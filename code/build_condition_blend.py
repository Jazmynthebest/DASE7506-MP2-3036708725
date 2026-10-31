"""Combine shared inference assets at the fixed train-CV-selected half weight."""
import argparse,json,shutil
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(); p.add_argument('--structure',default='checkpoints/condition')
    p.add_argument('--compact',default='checkpoints/compact_numpy'); p.add_argument('--output',default='checkpoints/condition_blend')
    args=p.parse_args(); source=ROOT/args.structure; compact=ROOT/args.compact; target=ROOT/args.output
    if target.exists(): raise FileExistsError('Do not overwrite checkpoint: '+str(target))
    # Both branches must use exactly the same first-stage models/calibration.
    for name in ('semantic_probes.npz','pair.npz','calibration.npz'):
        left=np.load(source/name); right=np.load(compact/name)
        assert set(left.files)==set(right.files)
        assert all(np.array_equal(left[k],right[k]) for k in left.files),name
    target.mkdir(parents=True)
    for name in ('semantic_probes.npz','pair.npz','calibration.npz','readout.npz','trees.npz'):
        shutil.copy2(source/name,target/name)
    shutil.copy2(compact/'trees.npz',target/'compact_trees.npz')
    config=dict(method='condition_blend',structure_weight=.5,rounds=120,leaves=7,l2=10,ridge_alpha=1000.,
        expanded_parser=True,selection='Fixed half blend selected on 3- and 5-fold train CV, before validation blend evaluation.')
    (target/'config.json').write_text(json.dumps(config,indent=2))
    print(json.dumps(dict(output=str(target),asset_bytes=sum(p.stat().st_size for p in target.iterdir()))))


if __name__=='__main__': main()
