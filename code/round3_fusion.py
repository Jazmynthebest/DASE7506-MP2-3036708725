"""Repeated training-only meta CV for preserving expert confidence in fusion."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='4'
import json,argparse
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from round2_features import answer_masks
from round2_nn import ranking_loss_gradient,pooled_logits
from round2_stack import fit_weights,normalized_scores
from train import ROOT,CACHE

OUT=ROOT/'runs'/'round3'; OLD=ROOT/'runs'/'round2'
SEEDS=[7511,7531,7551]


def normalize(scores,answers,epsilon):
    z,valid,_,_=pooled_logits(scores,answers)
    delta=np.where(valid,z-(np.where(valid,z,0).sum(1)/valid.sum(1))[:,None],0)
    if epsilon=='raw': return np.clip(delta,-8,8).astype(np.float32)
    return delta/np.sqrt((delta*delta).sum(1,keepdims=True)/valid.sum(1,keepdims=True)+epsilon**2)


def load_raw(inds,a,fixed=False):
    raw=[np.load((OUT/f'fixed_{name}_oof.npy') if fixed else (OLD/f'cv_{name}_oof.npy'))[inds]
         for name in ('semantic_ce_h15_l23','semantic_ce_h15_l21')]
    pairs=[np.load(OLD/f'cv_pair_raw64_l2_{r}_oof.npy')[inds] for r in (.001,.01)]
    raw+=[pairs,np.load(OLD/'v1_frozen_oof.npy')[inds]]
    return raw


def make_features(raw,a,epsilon=.25,combine_semantic=False):
    parts=[normalize(s,a,epsilon) for s in raw[:2]]
    if combine_semantic: parts=[np.mean(parts,axis=0)]
    parts+=[np.mean([normalize(s,a,epsilon) for s in raw[2]],axis=0),normalize(raw[3],a,epsilon)]
    valid,counts,_,_=answer_masks(a)
    consensus=np.log(np.maximum(counts,1))*valid
    consensus-=(consensus.sum(1,keepdims=True)/valid.sum(1,keepdims=True))*valid
    return np.stack(parts+[consensus],axis=2)


def fit_context(x,y,a,reg,shrink=1.):
    # Separate reliability for 2, 3, or 4 distinct answers, shrunk toward shared weights.
    valid,counts,_,_=answer_masks(a)
    context=np.rint((valid/np.maximum(counts,1)).sum(1)).astype(int)-2
    d=x.shape[-1]; weight=fit_weights(x,y,a,reg)
    def objective(flat):
        w=flat.reshape(3,d); mean=w.mean(0); delta=w-mean
        scores=np.einsum('ncd,nd->nc',x,w[context])
        loss,grad=ranking_loss_gradient(scores,y,a)
        gradient=reg*w/3+shrink*delta
        for k in range(3):
            selected=context==k
            gradient[k]+=np.einsum('ncd,nc->d',x[selected],grad[selected])
        return loss+reg*np.sum(w*w)/6+.5*shrink*np.sum(delta*delta),gradient.ravel()
    bounds=([(0,10)]*(d-1)+[(-2,2)])*3
    result=minimize(objective,np.tile(weight,(3,1)).ravel(),jac=True,method='L-BFGS-B',bounds=bounds,
                    options=dict(maxiter=150,ftol=1e-9))
    return result.x.reshape(3,d).astype(np.float32)


def predict_context(x,a,w):
    valid,counts,_,_=answer_masks(a)
    context=np.rint((valid/np.maximum(counts,1)).sum(1)).astype(int)-2
    return np.einsum('ncd,nd->nc',x,w[context])


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--fixed',action='store_true'); args=parser.parse_args()
    cache=np.load(CACHE,allow_pickle=False)
    y,a,hard=[cache['train_'+k] for k in ('y','answers','hard')]
    inds=np.flatnonzero(hard); y=y[inds]; a=a[inds]
    raw=load_raw(inds,a,fixed=args.fixed)
    strata=np.asarray([len(set(aa))*10+int(yy.sum()) for aa,yy in zip(a,y)])
    folds={seed:list(StratifiedKFold(n_splits=3,shuffle=True,random_state=seed).split(y,strata)) for seed in SEEDS}
    trials=[]
    for epsilon in (.25,.5,1.,2.,'raw'):
        for combine in (False,True):
            x=make_features(raw,a,epsilon,combine)
            for reg in (.01,.1):
                for shrink in (None, .1,1.):
                    corrects=[]; losses=[]
                    for seed in SEEDS:
                        scores=np.zeros((len(y),4),dtype=np.float32)
                        for ti,vi in folds[seed]:
                            if shrink is None:
                                w=fit_weights(x[ti],y[ti],a[ti],reg); scores[vi]=x[vi]@w
                            else:
                                w=fit_context(x[ti],y[ti],a[ti],reg,shrink)
                                scores[vi]=predict_context(x[vi],a[vi],w)
                        z,_,_,_=pooled_logits(scores,a); corrects.append(int(y[np.arange(len(y)),z.argmax(1)].sum()))
                        loss,_=ranking_loss_gradient(scores,y,a); losses.append(loss)
                    record=dict(epsilon=epsilon,combine_semantic=combine,regularization=reg,
                                shrink=shrink,corrects=corrects,mean_correct=float(np.mean(corrects)),
                                mean_accuracy=float(100*np.mean(corrects)/len(y)),mean_loss=float(np.mean(losses)))
                    trials.append(record)
            print('Finished epsilon '+str(epsilon)+' combine '+str(combine),flush=True)
    trials.sort(key=lambda t:(-t['mean_correct'],t['mean_loss']))
    report=dict(trials=trials,selected=trials[0],seeds=SEEDS,n=len(y),validation_read=False,test_read=False,
                note='Meta CV reuses base train OOF scores; development estimates, not an independent test.')
    report['fixed_production_tree_counts']=args.fixed
    (OUT/('fusion_fixed_selection.json' if args.fixed else 'fusion_selection.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(trials[:12],indent=2))


if __name__=='__main__': main()
