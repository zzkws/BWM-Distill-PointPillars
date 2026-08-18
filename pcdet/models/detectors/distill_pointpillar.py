"""
exp 2 -- feature distillation, CenterPoint-pillar (teacher) -> PointPillar (student).

Also the base class for exp 3 (consistency_distill_pointpillar.py). exp 3
subclasses this and only overrides get_extra_loss(), so the KD path is shared
code rather than a copy, and any exp2/exp3 delta is attributable to the extra
term alone.

Interface matches a plain detector, so it swaps in at the yaml level:
    __init__(model_cfg, num_class, dataset)
    forward() -> (ret_dict, tb_dict, disp_dict) when training, student_outs otherwise

Register in pcdet/models/detectors/__init__.py:
    from .distill_pointpillar import DistillPointPillar
    __all__['DistillPointPillar'] = DistillPointPillar

Assumptions
-----------
Teacher and student both use a dynamic pillar VFE that consumes raw points
(DynPillarVFE + transform_points_to_voxels_placeholder) and share the same
POINT_CLOUD_RANGE / VOXEL_SIZE. True for cbgs_dyn_pp_centerpoint.yaml paired
with the dyn-pillar student in the matching yaml.

Switching to a voxel-based teacher (cbgs_voxel01_res3d_centerpoint) breaks this
in two places: the teacher would need GPU re-voxelization of the points at its
own resolution, and a dataset proxy reporting its own grid_size so the sparse
backbone is built at the right spatial shape.

Changes from the earlier version of this file
---------------------------------------------
1. The teacher gets its own dict instead of the shared batch_dict. Previously
   the teacher forward overwrote pillar_features / voxel_coords /
   spatial_features_2d in place, so anything read from batch_dict afterwards
   was silently the teacher's.
2. The teacher stops at spatial_features_2d instead of running its head and NMS.
3. A 1x1 conv adapter sits in front of the student features.
4. Features are L2-normalized before the MSE, so KD_LOSS_WEIGHT is no longer
   fighting the raw activation scale. This changes the loss magnitude by orders
   of magnitude -- the old 10.0 does not carry over, and exp 2 has to be re-run.
5. Teacher weights are excluded from checkpoints and pinned to eval mode.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .detector3d_template import Detector3DTemplate
from .pointpillar import PointPillar
from .centerpoint import CenterPoint
from ...config import cfg_from_yaml_file
from easydict import EasyDict
from ...utils import common_utils
# exp 2


class DistillPointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg, num_class, dataset)
        # build_networks() is intentionally not called -- module_list stays
        # empty and self.student owns every trainable detector weight.

        # 1. Build student model
        self.student = PointPillar(model_cfg, num_class, dataset)

        # 2. Build teacher model
        teacher_cfg_path = model_cfg.TEACHER_CFG_FILE
        teacher_ckpt_path = model_cfg.TEACHER_CKPT

        logger = common_utils.create_logger(log_file=None, rank=0)
        logger.info(f"==> Loading Teacher Config from: {teacher_cfg_path}")

        teacher_cfg_root = EasyDict()
        cfg_from_yaml_file(teacher_cfg_path, teacher_cfg_root)

        # The teacher shares the student's dataset object, which is only valid
        # because both run at the same POINT_CLOUD_RANGE / VOXEL_SIZE.
        self.check_geometry(teacher_cfg_root, dataset, logger)
        self.teacher = CenterPoint(teacher_cfg_root.MODEL, num_class, dataset)

        logger.info(f"==> Loading Teacher Weights from: {teacher_ckpt_path}")
        self.teacher.load_params_from_file(filename=teacher_ckpt_path, logger=logger, to_cpu=False)

        # Freeze Teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        self.point_cloud_range = list(np.asarray(dataset.point_cloud_range).tolist())

        # 3. Channel adapter. Built here, not lazily on first forward, because
        #    OpenPCDet constructs the optimizer before any forward runs -- a
        #    lazily created layer would never receive gradients.
        #    With these two configs it is 384 -> 384; still worth keeping so the
        #    student is not forced to match the teacher's basis exactly.
        student_channels = self.student.backbone_2d.num_bev_features
        teacher_channels = self.teacher.backbone_2d.num_bev_features
        self.align = nn.Sequential(
            nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(teacher_channels),
            nn.ReLU(inplace=True)
        )

        # 4. Loss settings
        self.normalize_features = model_cfg.get('NORMALIZE_FEATURES', True)
        self.kd_loss_weight = model_cfg.get('KD_LOSS_WEIGHT', 1.0)
        self.warmup_steps = int(model_cfg.get('WARMUP_STEPS', 0))
        self.register_buffer('global_step_count', torch.zeros(1, dtype=torch.long), persistent=False)

    @staticmethod
    def check_geometry(teacher_cfg_root, dataset, logger):
        """Warn loudly if the teacher was trained on a different BEV geometry.

        Sharing the dataset object silently hands the teacher the student's
        grid_size, which is fine only when the two agree.
        """
        t_range = teacher_cfg_root.DATA_CONFIG.get('POINT_CLOUD_RANGE', None)
        if t_range is None:
            return
        s_range = np.asarray(dataset.point_cloud_range, dtype=np.float32)
        if not np.allclose(np.asarray(t_range, dtype=np.float32), s_range):
            logger.warning(f"==> Teacher POINT_CLOUD_RANGE {list(t_range)} != student "
                           f"{s_range.tolist()}; BEV features will not be spatially aligned.")

    # ------------------------------------------------------------------
    # Feature extraction and alignment. Plain methods so exp 3 and any
    # debugging script can reuse them.
    # ------------------------------------------------------------------

    @staticmethod
    def extract_bev_feature(model, batch_dict):
        """Run modules until spatial_features_2d appears, then stop.

        Skipping the dense head avoids target assignment and NMS entirely, and
        means a side branch never has to supply gt_boxes.
        """
        for cur_module in model.module_list:
            batch_dict = cur_module(batch_dict)
            if 'spatial_features_2d' in batch_dict:
                break
        return batch_dict['spatial_features_2d']

    def get_teacher_feature(self, points, batch_size):
        with torch.no_grad():
            self.teacher.eval()
            teacher_dict = {'batch_size': batch_size, 'points': points}
            return self.extract_bev_feature(self.teacher, teacher_dict).detach()

    def get_student_feature(self, points, batch_size):
        student_dict = {'batch_size': batch_size, 'points': points}
        return self.extract_bev_feature(self.student, student_dict)

    def align_features(self, student_feature, teacher_feature):
        """Match channels, then spatial size, then scale."""
        student_feature = self.align(student_feature)
        if student_feature.shape[-2:] != teacher_feature.shape[-2:]:
            teacher_feature = F.interpolate(teacher_feature, size=student_feature.shape[-2:],
                                            mode='bilinear', align_corners=False)
        if self.normalize_features:
            student_feature = F.normalize(student_feature, dim=1)
            teacher_feature = F.normalize(teacher_feature, dim=1)
        return student_feature, teacher_feature

    def get_loss_ramp(self):
        """Linear ramp so the distillation term does not dominate before the
        student's own head has taken shape."""
        if self.warmup_steps <= 0:
            return 1.0
        return float(min(1.0, self.global_step_count.item() / self.warmup_steps))

    def get_extra_loss(self, points, student_feature, teacher_feature, batch_size, ramp, tb_dict):
        """Hook for subclasses. exp 2 adds nothing beyond the KD term."""
        return 0.0

    # ------------------------------------------------------------------

    def forward(self, batch_dict):
        student_outs = self.student(batch_dict)

        if not self.training:
            return student_outs

        ret_dict, tb_dict, disp_dict = student_outs
        self.global_step_count += 1

        student_feature = batch_dict.get('spatial_features_2d', None)
        points = batch_dict.get('points', None)
        if student_feature is None or points is None:
            return ret_dict, tb_dict, disp_dict

        batch_size = batch_dict['batch_size']
        ramp = self.get_loss_ramp()

        # Teacher features on the original clouds
        teacher_feature = self.get_teacher_feature(points, batch_size)

        # Distillation loss
        s_aligned, t_aligned = self.align_features(student_feature, teacher_feature)
        kd_loss = F.mse_loss(s_aligned, t_aligned)
        distill_total_loss = kd_loss * self.kd_loss_weight * ramp
        tb_dict['kd_loss'] = kd_loss.item()

        distill_total_loss = distill_total_loss + self.get_extra_loss(
            points, student_feature, teacher_feature, batch_size, ramp, tb_dict)

        ret_dict['loss'] = ret_dict['loss'] + distill_total_loss
        tb_dict['distill_total'] = float(distill_total_loss)
        tb_dict['student_task_loss'] = ret_dict['loss'].item() - float(distill_total_loss)

        return ret_dict, tb_dict, disp_dict

    def get_training_loss(self):
        return self.student.get_training_loss()

    def train(self, mode=True):
        # the outer model.train() would otherwise flip the teacher's BN back
        # into training mode
        super().train(mode)
        self.teacher.eval()
        return self

    def state_dict(self, *args, **kwargs):
        # keep the frozen teacher out of checkpoints
        state = super().state_dict(*args, **kwargs)
        return {k: v for k, v in state.items() if not k.startswith('teacher.')}

    def load_state_dict(self, state_dict, strict=True):
        # teacher weights come from TEACHER_CKPT at build time, so they are
        # legitimately absent from any checkpoint written by this class
        return super().load_state_dict(state_dict, strict=False)
