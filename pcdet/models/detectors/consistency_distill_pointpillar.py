"""
exp 3 -- exp 2 plus Beam-Wise Mixing (BWM) consistency.

Subclasses DistillPointPillar and only overrides get_extra_loss(), so the
teacher construction, the BEV extraction, the adapter and the KD term are the
same code as exp 2. The consistency term is the sole difference between the two
experiments.

Register in pcdet/models/detectors/__init__.py:
    from .consistency_distill_pointpillar import ConsistencyDistillPP
    __all__['ConsistencyDistillPP'] = ConsistencyDistillPP

Why the mixing happens on the points, not the features
------------------------------------------------------
Consistency regularization means: perturb the input, and the output has to
follow the same perturbation. So the mixing operator must straddle the network.

Mixing student features and teacher features after both forwards with the same
mask M does nothing at all. M and (1-M) have disjoint support, so the cross
terms vanish; the batch roll cancels under the sum; and the loss reduces
exactly to the plain KD loss. That variant is a constant multiplier on
KD_LOSS_WEIGHT, nothing more.

Pipeline per training step (3 forwards)
---------------------------------------
    1) student(batch_dict)                 full forward -> detection loss + s_feat
    2) teacher BEV forward on raw points    -> t_feat            (no_grad, exp 2)
    3) student BEV forward on mixed points  -> s_feat_mix
       target = M * t_feat + (1-M) * t_feat[perm]

The mixed branch produces NO detection loss -- GT boxes straddling band
boundaries are not re-cut. BWM here is a feature-level regularizer, not
semi-supervised mixed-sample training.

Metrics to watch
----------------
    bwm_partner_ratio   fraction of points taken from the partner scene. Should
                        sit near 0.5. Far off means the bands are unbalanced and
                        little real mixing happens, in which case cons_loss just
                        converges onto kd_loss and exp 3 degenerates into exp 2.
    bwm_mask_mean       fraction of BEV *area* fed by the sample itself. NOT
                        expected to be 0.5: with uniform pitch bands the far
                        field collapses into a single band, so the area is
                        skewed even when the point split is balanced.
"""

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict
from .distill_pointpillar import DistillPointPillar
# exp 3


class ConsistencyDistillPP(DistillPointPillar):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg, num_class, dataset)

        bwm_cfg = model_cfg.get('BWM', EasyDict())
        self.bwm_enabled = bwm_cfg.get('ENABLED', True)
        self.num_areas_choices = list(bwm_cfg.get('NUM_AREAS', [3, 4, 5, 6]))
        pitch_lo, pitch_hi = bwm_cfg.get('PITCH_RANGE_DEG', [-30.0, 10.0])
        self.pitch_lo = float(np.deg2rad(pitch_lo))
        self.pitch_hi = float(np.deg2rad(pitch_hi))
        self.band_mode = bwm_cfg.get('BAND_MODE', 'uniform')
        self.ground_z = float(bwm_cfg.get('GROUND_Z', -1.8))
        self.mask_mode = bwm_cfg.get('MASK_MODE', 'raster')
        self.soft_mask = bwm_cfg.get('SOFT_MASK', True)
        self.cons_loss_weight = model_cfg.get('CONS_WEIGHT', 1.0)

    # ------------------------------------------------------------------
    # Beam-wise mixing. Plain methods so they can be called from outside
    # for debugging / visualization.
    # ------------------------------------------------------------------

    @staticmethod
    def compute_pitch(points):
        """theta = atan2(z, sqrt(x^2 + y^2)) per point, points in the
        (N, 1 + C) layout whose column 0 is the batch index.

        On a spinning LiDAR each laser sits at a fixed theta, so binning theta
        is effectively grouping beams -- this is where "beam-wise" comes from.

        Caveat for nuScenes: the 10-sweep aggregation motion-compensates past
        sweeps into the current frame, so their theta is no longer exactly the
        original beam angle. The partition stays geometrically sensible, it is
        just not a clean per-beam split.
        """
        radius = torch.norm(points[:, 1:3], dim=1).clamp(min=1e-6)
        return torch.atan2(points[:, 3], radius)

    def sample_band_edges(self, num_areas, device, theta=None):
        """Pitch band edges, with a random phase offset of up to half a band.

        Without the offset the boundaries always land on the same laser rings
        and the student can memorize which rows come from itself, which kills
        the regularization.

        uniform  -- equal width in theta, as in LaserMix. Simple, but on a
                    32-beam sensor returns cluster near the horizon, so the
                    bands carry very unequal point counts.
        quantile -- equal point count per band, keeping bwm_partner_ratio near
                    0.5 by construction.
        """
        if self.band_mode == 'quantile' and theta is not None:
            q = torch.linspace(0.0, 1.0, num_areas + 1, device=device)
            jitter = (torch.rand(1, device=device).item() - 0.5) / num_areas
            q = (q + jitter).clamp(0.0, 1.0)
            # torch.quantile has an input size limit, so subsample if needed
            sample = theta
            if theta.numel() > 1000000:
                sample = theta[torch.randperm(theta.numel(), device=device)[:1000000]]
            return torch.quantile(sample, q)

        edges = torch.linspace(self.pitch_lo, self.pitch_hi, num_areas + 1, device=device)
        step = (self.pitch_hi - self.pitch_lo) / num_areas
        return edges + (torch.rand(1, device=device).item() - 0.5) * step

    @staticmethod
    def assign_bands(theta, edges):
        """Band index per point.

        Bucketizing on edges[1:-1] lets out-of-range points fall into the end
        bands instead of being dropped, so the point count is conserved.
        """
        band = torch.bucketize(theta, edges[1:-1].contiguous())
        return band.clamp_(0, edges.numel() - 2)

    def mix_points(self, points, batch_size, edges, theta):
        """Beam-wise mixing on the raw cloud.

        Convention: sample b keeps its own even bands and takes the odd bands
        of partner perm[b].

        perm is a cyclic shift, which guarantees perm[b] != b (no sample mixes
        with itself) and has the closed-form inverse perm^-1[j] = (j - shift) % B.
        That inverse is what the relabeling uses: a point currently in sample j
        with odd parity belongs to whichever sample has j as its partner.

        Returns:
            mixed_points: (N, 1 + C), same layout and same total point count.
            source_flag:  (N,) 0 = kept from itself, 1 = taken from the partner.
            perm:         (B,) partner index per sample.
        """
        device = points.device
        batch_idx = points[:, 0].long()
        parity = self.assign_bands(theta, edges) % 2

        shift = int(torch.randint(1, batch_size, (1,)).item())
        perm = (torch.arange(batch_size, device=device) + shift) % batch_size

        keep_self = parity == 0
        pts_self = points[keep_self]
        pts_partner = points[~keep_self].clone()
        pts_partner[:, 0] = ((batch_idx[~keep_self] - shift) % batch_size).to(points.dtype)

        mixed_points = torch.cat([pts_self, pts_partner], dim=0)
        source_flag = torch.cat([
            torch.zeros(pts_self.shape[0], device=device),
            torch.ones(pts_partner.shape[0], device=device)
        ], dim=0)
        return mixed_points, source_flag, perm

    def analytic_ring_mask(self, edges, feat_shape, device):
        """Pitch bands projected onto BEV under a flat-ground assumption.

        For a ground point z = ground_z, so r = ground_z / tan(theta): equal
        theta bands become concentric annuli around the sensor. Rings, not
        horizontal stripes -- stripes in the Cartesian BEV grid have no relation
        to laser beams at all.

        Only used to fill BEV cells that no point lands in.
        """
        H, W = feat_shape
        x_min, y_min, _, x_max, y_max, _ = self.point_cloud_range
        xs = x_min + (torch.arange(W, device=device).float() + 0.5) * (x_max - x_min) / W
        ys = y_min + (torch.arange(H, device=device).float() + 0.5) * (y_max - y_min) / H
        radius = torch.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2)

        # theta < 0 is below the horizon; at or above it the ring runs past max range
        tan_theta = torch.tan(edges)
        r_edges = torch.where(tan_theta < -1e-4, self.ground_z / tan_theta,
                              torch.full_like(tan_theta, 1e6))
        r_edges = torch.cummax(r_edges, dim=0)[0]  # enforce monotonicity

        band = torch.bucketize(radius.reshape(-1), r_edges[1:-1].contiguous())
        band = band.clamp_(0, edges.numel() - 2).reshape(H, W)
        return (band % 2 == 0).float()[None, None]

    def build_mix_mask(self, mixed_points, source_flag, batch_size, feat_shape, edges):
        """BEV mask saying which teacher each cell's target comes from.

        Rasterized from where the mixed points actually landed rather than from
        the analytic rings, because the flat-ground relation breaks for points
        on tall structures -- a car roof and the road at the same radius have
        noticeably different pitch and can land in different bands.

        Cell size is derived from feat_shape at runtime, so no backbone stride
        is hardcoded. Empty cells fall back to the analytic ring value. Soft
        mode keeps the fractional value at band boundaries, which is a better
        regression target than a hard 0/1 edge.
        """
        device = mixed_points.device
        fallback = self.analytic_ring_mask(edges, feat_shape, device)
        if self.mask_mode != 'raster':
            return fallback.expand(batch_size, 1, *feat_shape)

        H, W = feat_shape
        x_min, y_min, _, x_max, y_max, _ = self.point_cloud_range
        b = mixed_points[:, 0].long()
        ix = ((mixed_points[:, 1] - x_min) / ((x_max - x_min) / W)).long()
        iy = ((mixed_points[:, 2] - y_min) / ((y_max - y_min) / H)).long()
        valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        flat = b[valid] * H * W + iy[valid] * W + ix[valid]

        n_self = torch.zeros(batch_size * H * W, device=device)
        n_partner = torch.zeros(batch_size * H * W, device=device)
        flag = source_flag[valid]
        n_self.scatter_add_(0, flat, 1.0 - flag)
        n_partner.scatter_add_(0, flat, flag)

        total = n_self + n_partner
        mask = torch.where(total > 0, n_self / total.clamp(min=1.0),
                           fallback.expand(batch_size, 1, H, W).reshape(-1))
        mask = mask.view(batch_size, 1, H, W)
        return mask if self.soft_mask else (mask > 0.5).float()

    # ------------------------------------------------------------------

    def get_extra_loss(self, points, student_feature, teacher_feature, batch_size, ramp, tb_dict):
        """The consistency term. The student forward sits between the two halves
        of the mixing operator, which is what stops this collapsing into
        kd_loss."""
        if not self.bwm_enabled or batch_size < 2:
            return 0.0

        device = student_feature.device
        feat_shape = tuple(student_feature.shape[-2:])

        num_areas = int(np.random.choice(self.num_areas_choices))
        theta = self.compute_pitch(points)
        edges = self.sample_band_edges(num_areas, device, theta)

        mixed_points, source_flag, perm = self.mix_points(points, batch_size, edges, theta)
        mixed_student_feature = self.get_student_feature(mixed_points, batch_size)
        # same edges as the mixing above -- resampling here would desync the
        # mask from the actual point partition
        mask = self.build_mix_mask(mixed_points, source_flag, batch_size, feat_shape, edges)

        with torch.no_grad():
            teacher_ref = teacher_feature
            if teacher_ref.shape[-2:] != feat_shape:
                teacher_ref = F.interpolate(teacher_ref, size=feat_shape,
                                            mode='bilinear', align_corners=False)
            # row b takes teacher_ref[perm[b]], matching the point relabeling
            mixed_target = mask * teacher_ref + (1.0 - mask) * teacher_ref[perm]

        s_mix, t_mix = self.align_features(mixed_student_feature, mixed_target)
        cons_loss = F.mse_loss(s_mix, t_mix)

        tb_dict['cons_loss'] = cons_loss.item()
        tb_dict['bwm_num_areas'] = float(num_areas)
        tb_dict['bwm_partner_ratio'] = source_flag.mean().item()
        tb_dict['bwm_mask_mean'] = mask.mean().item()

        return cons_loss * self.cons_loss_weight * ramp
