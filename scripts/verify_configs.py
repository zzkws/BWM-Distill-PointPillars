#!/usr/bin/env python3
"""Verify that the three experiment configs differ in exactly one variable.

The whole point of a three-arm ablation is that the arms are identical apart
from the thing being ablated. This script checks that claim mechanically
instead of leaving it to inspection: it flattens all three YAML files to
dot-paths and reports any key that differs outside the whitelist below.

Usage:
    python scripts/verify_configs.py
    python scripts/verify_configs.py --cfg-dir tools/cfgs/nuscenes_models

Exit code 0 = the arms are comparable, 1 = a confound was found.
"""

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("PyYAML is required: pip install pyyaml")

# Keys that are *supposed* to differ. Everything else must match across arms.
ALLOWED_PREFIXES = (
    'MODEL.NAME',              # the detector class being selected
    'MODEL.TEACHER_CFG_FILE',  # exp 2 / exp 3 only
    'MODEL.TEACHER_CKPT',
    'MODEL.NORMALIZE_FEATURES',
    'MODEL.KD_LOSS_WEIGHT',
    'MODEL.WARMUP_STEPS',
    'MODEL.CONS_WEIGHT',       # exp 3 only
    'MODEL.BWM',
)

EXPERIMENTS = {
    'exp1_baseline': 'pointpillar.yaml',
    'exp2_distill': 'distill_pointpillar.yaml',
    'exp3_distill_bwm': 'consistency_distill_pointpillar.yaml',
}


def flatten(node, prefix=''):
    """Flatten nested dicts to {dot.path: value}. Lists are compared whole."""
    flat = {}
    if isinstance(node, dict):
        for key, value in node.items():
            flat.update(flatten(value, f'{prefix}.{key}' if prefix else str(key)))
    elif isinstance(node, list) and any(isinstance(item, dict) for item in node):
        for i, item in enumerate(node):
            flat.update(flatten(item, f'{prefix}[{i}]'))
    else:
        flat[prefix] = node
    return flat


def is_allowed(key):
    return any(key == p or key.startswith(p + '.') or key.startswith(p + '[')
               for p in ALLOWED_PREFIXES)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cfg-dir', default='tools/cfgs/nuscenes_models',
                        help='directory holding the three experiment YAMLs')
    args = parser.parse_args()

    cfg_dir = Path(args.cfg_dir)
    configs = {}
    for name, filename in EXPERIMENTS.items():
        path = cfg_dir / filename
        if not path.exists():
            sys.exit(f'missing config: {path}')
        with path.open(encoding='utf-8') as handle:
            configs[name] = flatten(yaml.safe_load(handle))
        print(f'loaded {name:18s} {path}  ({len(configs[name])} keys)')

    all_keys = set()
    for flat in configs.values():
        all_keys.update(flat)

    confounds = []
    for key in sorted(all_keys):
        if is_allowed(key):
            continue
        values = {name: flat.get(key, '<absent>') for name, flat in configs.items()}
        if len(set(map(repr, values.values()))) > 1:
            confounds.append((key, values))

    print()
    if not confounds:
        print('PASS - the three arms are key-for-key identical outside the '
              'distillation / BWM blocks.')
        print('       Any exp1 -> exp2 -> exp3 delta is attributable to the '
              'loss terms alone.')
        return 0

    print(f'FAIL - {len(confounds)} confounding key(s) found:\n')
    for key, values in confounds:
        print(f'  {key}')
        for name, value in values.items():
            print(f'      {name:18s} {value!r}')
        print()
    print('Each of these differs between arms while not being part of the '
          'ablation. Fix them before reporting a comparison.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
