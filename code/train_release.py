"""Train the fixed final architecture from raw training inputs only."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('checkpoints/retrained'));args=p.parse_args()
    target=(ROOT/args.output).resolve()
    if not target.is_relative_to(ROOT) or target.exists():raise ValueError('Output must be a new path inside code/.')
    work=ROOT/'runs'/'training_reproduction_work'
    if work.exists():raise FileExistsError('Use a clean working copy for training reproduction.')
    work.mkdir(parents=True);(ROOT/'runs/round6').mkdir(exist_ok=True)
    env=dict(os.environ)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):env[key]='4'
    rel=lambda p:str(p.relative_to(ROOT))
    commands=[['train_condition.py','--output',rel(work/'structure')],
        ['train_compact.py','--output',rel(work/'compact_raw')],
        ['convert_compact_numpy.py','--source',rel(work/'compact_raw'),'--output',rel(work/'compact_numpy')],
        ['build_condition_blend.py','--structure',rel(work/'structure'),'--compact',rel(work/'compact_numpy'),'--output',rel(target)]]
    timings=[]
    for command in commands:
        start=time.perf_counter();subprocess.run([sys.executable,*command],cwd=ROOT,env=env,check=True)
        timings.append(dict(command=command,seconds=time.perf_counter()-start))
    import numpy as np
    comparisons={}
    for path in (ROOT/'checkpoints/condition_blend').glob('*.npz'):
        with np.load(path,allow_pickle=False) as a,np.load(target/path.name,allow_pickle=False) as b:
            assert set(a.files)==set(b.files),path.name
            error=max(float(np.abs(a[k]-b[k]).max(initial=0)) for k in a.files)
            comparisons[path.name]=error
            assert error<1e-5,(path.name,error)
    assert json.loads((target/'config.json').read_text())==json.loads((ROOT/'checkpoints/condition_blend/config.json').read_text())
    result=dict(status='passed',timings=timings,total_process_seconds=sum(r['seconds'] for r in timings),
        parameter_max_differences=comparisons,seed=7506,validation_read=False,test_read=False)
    (ROOT/'runs/training_reproduction.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
