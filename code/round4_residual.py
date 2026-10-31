"""Cross-fitted linear verifier followed by explicit residual gradient boosting."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='4'
import json,time
import numpy as np
import lightgbm as lgb
from scipy.special import logsumexp
from sklearn.model_selection import KFold
from optimize_round2_linear import fit_pairwise
from round2_features import answer_masks
from round2_stack import fit_weights
from round3_fusion import normalize
from round2_nn import pooled_logits,ranking_loss_gradient
from train import ROOT,CACHE

OUT=ROOT/'runs'/'round4'; OLD=ROOT/'runs'/'round2'
ITERATIONS=[40,80,120,180,260,400]


def initial_features(full_pair,semantic,answers):
    parts=[normalize(full_pair,answers,1.),normalize(semantic[:,:,284],answers,1.),normalize(semantic[:,:,289],answers,1.)]
    valid,counts,_,_=answer_masks(answers)
    consensus=np.log(np.maximum(counts,1))*valid
    consensus-=(consensus.sum(1,keepdims=True)/valid.sum(1,keepdims=True))*valid
    return np.stack(parts+[consensus],axis=2)


def fit_residual(tx,y,a,initial,config,rounds=400):
    valid,counts,mean,_=answer_masks(a)
    target=y/y.sum(1,keepdims=True)
    def objective(pred,dataset):
        logits=pred.reshape(-1,4); scores=(mean@logits[:,:,None])[:,:,0]
        adjusted=np.where(valid,scores-np.log(np.maximum(counts,1)),-1e4)
        p=np.exp(adjusted-logsumexp(adjusted,axis=1,keepdims=True))
        grad=(mean.transpose(0,2,1)@(p-target)[:,:,None])[:,:,0]
        expected=(mean.transpose(0,2,1)@p[:,:,None])[:,:,0]
        hessian=((mean*mean).transpose(0,2,1)@p[:,:,None])[:,:,0]-expected*expected
        return grad.ravel(),np.maximum(hessian,1e-6).ravel()
    params=dict(objective=objective,metric='None',learning_rate=.025,num_leaves=config.get('leaves',7),
                min_data_in_leaf=config.get('min_leaf',100),lambda_l2=config.get('l2',10),
                feature_fraction=.8,bagging_fraction=.85,bagging_freq=1,num_threads=4,
                verbosity=-1,seed=2026,deterministic=True,force_col_wise=True,max_bin=63)
    data=lgb.Dataset(tx.reshape(-1,tx.shape[-1]),label=y.ravel(),group=[4]*len(y),init_score=initial.ravel())
    return lgb.train(params,data,num_boost_round=rounds)


def select_features(base,semantic,kind):
    if kind=='semantic': return semantic
    if kind=='raw': return np.concatenate([base,semantic[:,:,284:]],axis=2)
    if kind=='numeric':
        raise ValueError('Numeric cache is added by the caller')
    return base[:,:,2048:]


def main():
    OUT.mkdir(exist_ok=True)
    cache=np.load(CACHE,allow_pickle=False)
    x,y,a,hard=[cache['train_'+k] for k in ('x','y','answers','hard')]
    inds=np.flatnonzero(hard)
    outer_pair=np.load(OLD/'cv_pair_raw64_l2_0.01_oof.npy')
    for fold in range(3):
        path=OUT/f'residual_initial_{fold}.npz'
        if path.exists(): continue
        data=np.load(OLD/f'semantic_nested_features_{fold}.npz',allow_pickle=False)
        ti,vi=data['train_indices'],data['validation_indices']
        scores=np.zeros((len(ti),4),dtype=np.float32)
        for it,iv in KFold(n_splits=3,shuffle=True,random_state=7509).split(ti):
            _,scores[iv],_=fit_pairwise(x[ti[it]],y[ti[it]],a[ti[it]],x[ti[iv]],regularization=.01)
        tx=initial_features(scores,data['train_features'],a[ti])
        vx=initial_features(outer_pair[vi],data['validation_features'],a[vi])
        weights=fit_weights(tx,y[ti],a[ti],.01)
        np.savez(path,train_initial=tx@weights,validation_initial=vx@weights,weights=weights,
                 train_pair=scores,validation_pair=outer_pair[vi],train_indices=ti,validation_indices=vi)
        print('Nested residual initial scores ready fold '+str(fold),flush=True)
    configs=[dict(kind=kind,leaves=leaves,l2=l2) for kind in ('semantic','raw','text') for leaves,l2 in ((7,10),(15,10),(7,30))]
    for config in configs:
        name=f'residual_{config["kind"]}_{config["leaves"]}_l2{config["l2"]}'
        if (OUT/f'cv_{name}.json').exists(): continue
        curves=np.zeros((len(ITERATIONS),len(x),4),dtype=np.float32); start=time.perf_counter()
        for fold in range(3):
            semantic=np.load(OLD/f'semantic_nested_features_{fold}.npz',allow_pickle=False)
            initial=np.load(OUT/f'residual_initial_{fold}.npz',allow_pickle=False)
            ti,vi=initial['train_indices'],initial['validation_indices']
            tx=select_features(x[ti],semantic['train_features'],config['kind'])
            vx=select_features(x[vi],semantic['validation_features'],config['kind'])
            model=fit_residual(tx,y[ti],a[ti],initial['train_initial'],config)
            for j,iteration in enumerate(ITERATIONS):
                curves[j,vi]=model.predict(vx.reshape(-1,vx.shape[-1]),raw_score=True,num_threads=4,
                                           num_iteration=iteration).reshape(-1,4)+initial['validation_initial']
        trials=[]
        for iteration,scores in zip(ITERATIONS,curves):
            z,_,_,_=pooled_logits(scores[inds],a[inds]); chosen=z.argmax(1)
            loss,_=ranking_loss_gradient(scores[inds],y[inds],a[inds])
            trials.append(dict(iteration=iteration,correct=int(y[inds,chosen].sum()),loss=loss))
        result=dict(name=name,config=config,n=len(inds),trials=trials,seconds=time.perf_counter()-start,
                    validation_read=False,test_read=False)
        np.savez(OUT/f'cv_{name}_curves.npz',iterations=ITERATIONS,scores=curves)
        (OUT/f'cv_{name}.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)


if __name__=='__main__': main()
