"""Fresh prediction plus independent scoring; four threads per process."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=('validation','test'),default='validation')
    p.add_argument('--output',type=Path);args=p.parse_args()
    output=args.output or ROOT/'runs'/('reproduced-'+args.split);output.mkdir(parents=True,exist_ok=True)
    predictions=output/'predictions.json';score=output/'score.json';env=dict(os.environ)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):env[key]='4'
    commands=[[sys.executable,str(ROOT/'predict.py'),'--method',str(ROOT/'submission.py'),
        '--checkpoint',str(ROOT/'checkpoints/condition_blend'),'--split',args.split,'--output',str(predictions)],
        [sys.executable,str(ROOT/'score_predictions.py'),'--split',args.split,'--predictions',str(predictions),'--output',str(score)]]
    times=[]
    for command in commands:
        start=time.perf_counter();subprocess.run(command,cwd=ROOT,env=env,check=True);times.append(time.perf_counter()-start)
    result=dict(split=args.split,score=json.loads(score.read_text()),prediction_wall_seconds=times[0],
        scoring_wall_seconds=times[1],combined_wall_seconds=sum(times),threads=4,
        asset_bytes=sum(p.stat().st_size for p in (ROOT/'checkpoints/condition_blend').iterdir()))
    (output/'resources.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
