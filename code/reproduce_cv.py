"""Optional nested training CV reproduction from supplied raw data."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
import argparse,json,time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from common import load_inputs,read_jsonl,parse_answer
from train import ROOT,labels_for
from features import make_features
from structure_math import extract_program,program_vector
from round6_parser import chained_program
from round10_pipeline import train_member
from submission_compact import answer_groups
from numpy_trees import predict_trees
def main():
    p=argparse.ArgumentParser();p.add_argument('--folds',type=int,choices=(3,5),default=3);args=p.parse_args()
    out=ROOT/'runs'/('cv_reproduction_'+str(args.folds));out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    bank,arrays=load_inputs('train');x=make_features(bank,arrays,'full');y=labels_for(bank,'train')
    a=np.asarray([['INVALID' if (s:=parse_answer(c['text'])) is None else s for c in r['candidates']] for r in bank])
    hard=np.asarray([len(set(aa)-{'INVALID'})>=2 for aa in a]);inds=np.flatnonzero(hard);easy=np.flatnonzero(~hard)
    refs=read_jsonl(ROOT/'data/gsm8k_train.jsonl');targets=[];mask=[];actual=[];parsed=[]
    for r,ref in zip(bank,refs):
        teacher=extract_program(r['question'],ref['answer']);targets.append(program_vector(teacher));mask.append(teacher is not None)
        ps=[chained_program(r['question'],c['text']) for c in r['candidates']]
        actual.append([program_vector(s) for s in ps]);parsed.append([s is not None for s in ps])
    targets=np.asarray(targets);mask=np.asarray(mask);actual=np.asarray(actual);parsed=np.asarray(parsed)
    strata=np.asarray([len(set(aa))*10+int(yy.sum()) for aa,yy in zip(a[inds],y[inds])]);records=[]
    scores={k:np.zeros((len(y),4)) for k in ('compact','arithmetic','blend')}
    splitter=StratifiedKFold(n_splits=args.folds,shuffle=True,random_state=7506 if args.folds==3 else 7591)
    for fold,(tr,va) in enumerate(splitter.split(inds,strata)):
        ti,vi=inds[tr],inds[va];target=out/('fold'+str(fold))
        blended,record,data,initial,vinitial=train_member(x,y,a,targets,mask,actual,parsed,ti,vi,easy,np.ones(len(y),dtype=np.int64),target)
        scores['blend'][vi]=blended
        for name,width,filename in [('compact',294,'compact_trees.npz'),('arithmetic',387,'structure_trees.npz')]:
            with np.load(target/filename,allow_pickle=False) as f:trees={k:f[k] for k in f.files}
            vf=data['vf'][:,:,:width]
            scores[name][vi]=predict_trees(vf.reshape(-1,width),trees).reshape(-1,4)+vinitial
        records.append(record);print('Completed outer fold '+str(fold),flush=True)
    valid,_,own,_=answer_groups(a);metrics={}
    for name,z in scores.items():
        grouped=np.where(valid,(own@z[:,:,None])[:,:,0],-1e4);choice=grouped[inds].argmax(1)
        metrics[name]=int(y[inds,choice].sum());np.save(out/(name+'_oof.npy'),z)
    result=dict(folds=args.folds,correct=metrics,n=len(inds),seconds=time.perf_counter()-started,fold_records=records,
        validation_read=False,test_read=False)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
