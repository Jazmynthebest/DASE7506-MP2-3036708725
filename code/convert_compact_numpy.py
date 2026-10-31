"""Export an independently trained compact model to equivalent NumPy assets."""
import argparse,json,shutil
from pathlib import Path
from export_numpy_trees import export_trees

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--source',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); source=ROOT/args.source; output=ROOT/args.output
    if output.exists(): raise FileExistsError('Do not overwrite a checkpoint: '+str(output))
    output.mkdir(parents=True)
    for name in ('semantic_probes.npz','pair.npz','calibration.npz'): shutil.copyfile(source/name,output/name)
    export_trees(source/'residual.txt',output/'trees.npz')
    config=dict(method='compact_numpy',rounds=120,tree_inference='NumPy, numeric finite features, float64 comparisons',
                teacher_program_readout=False,ridge_alpha=None)
    (output/'config.json').write_text(json.dumps(config,indent=2))


if __name__=='__main__': main()
