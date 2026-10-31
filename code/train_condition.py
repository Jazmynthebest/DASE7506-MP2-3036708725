"""Reproduce the expanded arithmetic-chain verifier from supplied raw training data only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='4'
import argparse,json,time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from common import load_inputs,parse_answer,read_jsonl
from features import make_features
from train import ROOT,labels_for
from structure_math import extract_program,program_vector,compatibility
from round6_parser import chained_program as candidate_program
from round5_program_readout import fit_readout,readout
from round2_semantic_stack import fit_probes,probe_scores,small_features
from optimize_round2_linear import fit_pairwise
from round4_residual import initial_features,fit_residual
from round2_stack import fit_weights
from export_numpy_trees import export_trees


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='checkpoints/condition_retrained'); args=parser.parse_args()
    target=ROOT/args.output
    if target.exists(): raise FileExistsError('Do not overwrite a checkpoint: '+str(target))
    start=time.perf_counter(); bank,arrays=load_inputs('train')
    x=make_features(bank,arrays,'full'); y=labels_for(bank,'train')
    a=np.asarray([['INVALID' if (s:=parse_answer(c['text'])) is None else s for c in row['candidates']] for row in bank])
    hard=np.asarray([len(set(aa)-{'INVALID'})>=2 for aa in a]); inds=np.flatnonzero(hard); easy=np.flatnonzero(~hard)
    refs=read_jsonl(ROOT/'data'/'gsm8k_train.jsonl')
    targets=[]; target_mask=[]; actual=[]; parsed=[]
    for row,ref in zip(bank,refs):
        assert row['id']==ref['id']
        program=extract_program(row['question'],ref['answer']); targets.append(program_vector(program)); target_mask.append(program is not None)
        programs=[candidate_program(row['question'],c['text']) for c in row['candidates']]
        actual.append([program_vector(p) for p in programs]); parsed.append([p is not None for p in programs])
    targets=np.asarray(targets); target_mask=np.asarray(target_mask); actual=np.asarray(actual); parsed=np.asarray(parsed)
    tf=np.zeros((len(x),4,387),dtype=np.float32); pair=np.zeros((len(x),4),dtype=np.float32)
    strata=np.asarray([len(set(aa))*10+int(yy.sum()) for aa,yy in zip(a[inds],y[inds])])
    for fold,(tr,va) in enumerate(StratifiedKFold(n_splits=3,shuffle=True,random_state=7506).split(inds,strata)):
        ti,vi=inds[tr],inds[va]; fitting=np.r_[ti,easy]
        probes=fit_probes(x[fitting],y[fitting],a[fitting]); semantic=small_features(x[vi],a[vi],probe_scores(x[vi],probes))
        learned=fit_readout(x[fitting],targets[fitting],target_mask[fitting],1000.)
        tf[vi]=np.concatenate([semantic,compatibility(actual[vi],parsed[vi],readout(x[vi],learned))],axis=2)
        _,pair[vi],_=fit_pairwise(x[ti],y[ti],a[ti],x[vi],regularization=.01)
        print('Structural training-only cross-fitting fold '+str(fold)+' complete',flush=True)
    calibration=fit_weights(initial_features(pair[inds],tf[inds,:,:294],a[inds]),y[inds],a[inds],.01)
    initial=initial_features(pair[inds],tf[inds,:,:294],a[inds])@calibration
    residual=fit_residual(tf[inds],y[inds],a[inds],initial,dict(leaves=7,l2=10),rounds=120)
    learned=fit_readout(x,targets,target_mask,1000.); probes=fit_probes(x,y,a)
    pair_model,_,_=fit_pairwise(x,y,a,x[:1],regularization=.01)
    target.mkdir(parents=True)
    np.savez(target/'semantic_probes.npz',**probes); np.savez(target/'pair.npz',weight=pair_model['weight'])
    np.savez(target/'calibration.npz',weight=calibration); np.savez(target/'readout.npz',**learned)
    export_trees(residual,target/'trees.npz')
    config=dict(method='condition',rounds=120,tree_inference='NumPy, numeric finite features, float64 comparisons',
                teacher_program_readout=True,ridge_alpha=1000.,expanded_parser=True)
    (target/'config.json').write_text(json.dumps(config,indent=2))
    result=dict(output=str(target),seconds=time.perf_counter()-start,seed=7506,teacher_programs=int(target_mask.sum()),
                n=len(x),asset_bytes=sum(p.stat().st_size for p in target.iterdir()),validation_read=False,test_read=False)
    (ROOT/'runs/round6/condition_retraining.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))


if __name__=='__main__': main()

