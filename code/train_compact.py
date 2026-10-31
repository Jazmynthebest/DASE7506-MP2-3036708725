"""Reproduce compact checkpoint from training inputs and references only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='4'
import argparse,json,time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from common import load_inputs,parse_answer
from features import make_features
from train import ROOT,CACHE,labels_for
from round2_semantic_stack import fit_probes,probe_scores,small_features
from optimize_round2_linear import fit_pairwise
from round2_stack import fit_weights
from round4_residual import fit_residual,initial_features


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=str,default='checkpoints/compact')
    parser.add_argument('--reuse-cache',action='store_true'); args=parser.parse_args()
    target=ROOT/args.output
    if target.exists(): raise FileExistsError('Do not overwrite a frozen checkpoint: '+str(target))
    start=time.perf_counter()
    if args.reuse_cache:
        c=np.load(CACHE,allow_pickle=False); x,y,a,hard=[c['train_'+k] for k in ('x','y','answers','hard')]
    else:
        bank,arrays=load_inputs('train'); x=make_features(bank,arrays,'full'); y=labels_for(bank,'train')
        a=np.asarray([['INVALID' if (v:=parse_answer(c['text'])) is None else v for c in r['candidates']] for r in bank])
        hard=np.asarray([len(set(aa)-{'INVALID'})>=2 for aa in a])
    inds=np.flatnonzero(hard); easy=np.flatnonzero(~hard)
    strata=np.asarray([len(set(aa))*10+int(yy.sum()) for aa,yy in zip(a[inds],y[inds])])
    features=np.zeros((len(x),4,294),dtype=np.float32); pair=np.zeros((len(x),4),dtype=np.float32)
    for fold,(tr,va) in enumerate(StratifiedKFold(n_splits=3,shuffle=True,random_state=7506).split(inds,strata)):
        ti,vi=inds[tr],inds[va]; fitting=np.r_[ti,easy]
        probes=fit_probes(x[fitting],y[fitting],a[fitting])
        features[vi]=small_features(x[vi],a[vi],probe_scores(x[vi],probes))
        _,pair[vi],_=fit_pairwise(x[ti],y[ti],a[ti],x[vi],regularization=.01)
        print('Training-only cross-fitting fold '+str(fold)+' complete',flush=True)
    calibration=fit_weights(initial_features(pair[inds],features[inds],a[inds]),y[inds],a[inds],.01)
    initial=initial_features(pair[inds],features[inds],a[inds])@calibration
    residual=fit_residual(features[inds],y[inds],a[inds],initial,dict(leaves=7,l2=10),rounds=120)
    full_probes=fit_probes(x,y,a)
    full_pair,_,_=fit_pairwise(x,y,a,x[:1],regularization=.01)
    target.mkdir(parents=True)
    np.savez(target/'semantic_probes.npz',**full_probes)
    np.savez(target/'pair.npz',weight=full_pair['weight'])
    np.savez(target/'calibration.npz',weight=calibration)
    residual.save_model(str(target/'residual.txt'))
    config=dict(version=1,method='cross-fitted linear verifier with residual answer-group boosting',
                rounds=120,leaves=7,l2=10,calibration_regularization=.01,fold_seed=7506,
                training_data='supplied training inputs/references only',inference_threads=4)
    (target/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    result=dict(checkpoint=str(target),seconds=time.perf_counter()-start,
                asset_bytes=sum(p.stat().st_size for p in target.iterdir()),validation_read=False,test_read=False,
                calibration_weights=calibration.tolist(),seed=7506)
    out=ROOT/'runs'/'round4'; out.mkdir(exist_ok=True)
    (out/'compact_training.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
