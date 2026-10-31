"""Compare task-aligned ranking objectives with groupwise train-only CV."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='4'
import json
import time
import numpy as np
import lightgbm as lgb
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedKFold
from round2_features import answer_masks
from round2_nn import pooled_logits,ranking_loss_gradient
from train import ROOT,CACHE

OUT=ROOT/'runs'/'round2'


def fit_ranking(tx,y,a,vx,vy,va,config,rounds=None):
    params={'objective':config['objective'],'metric':'None','learning_rate':0.035,
            'num_leaves':config['leaves'],'min_data_in_leaf':60 if config['leaves']==15 else 80,
            'lambda_l2':config.get('l2',5),'feature_fraction':0.8,'bagging_fraction':0.85,
            'bagging_freq':1,'num_threads':4,'verbosity':-1,'seed':2026,
            'deterministic':True,'force_col_wise':True,'max_bin':63}
    if config['objective']=='answer_ce':
        valid,counts,mean,_=answer_masks(a)
        def custom_objective(pred,dataset):
            logits=pred.reshape(-1,4)
            scores=(mean @ logits[:,:,None])[:,:,0]
            adjusted=np.where(valid,scores-np.log(np.maximum(counts,1)),-1e4)
            p=np.exp(adjusted-logsumexp(adjusted,axis=1,keepdims=True))
            target=y/y.sum(1,keepdims=True)
            grad=(mean.transpose(0,2,1) @ (p-target)[:,:,None])[:,:,0]
            expected=(mean.transpose(0,2,1) @ p[:,:,None])[:,:,0]
            hessian=((mean*mean).transpose(0,2,1) @ p[:,:,None])[:,:,0]-expected*expected
            return grad.ravel(),np.maximum(hessian,1e-6).ravel()
        params['objective']=custom_objective
    def evaluate(pred,dataset):
        scores=pred.reshape(-1,4)
        pooled,_,_,_=pooled_logits(scores,va)
        selected=pooled.argmax(1)
        acc=float(100*vy[np.arange(len(vy)),selected].mean())
        return 'answer_selection_accuracy',acc,True
    train=lgb.Dataset(tx.reshape(-1,tx.shape[-1]),label=y.ravel(),group=[4]*len(y))
    valid=lgb.Dataset(vx.reshape(-1,vx.shape[-1]),label=vy.ravel(),group=[4]*len(vy),reference=train)
    start=time.perf_counter()
    model=lgb.train(params,train,num_boost_round=rounds or 800,valid_sets=[valid],feval=evaluate,
                    callbacks=[] if rounds else [lgb.early_stopping(100,verbose=False)])
    scores=model.predict(vx.reshape(-1,vx.shape[-1]),raw_score=True,num_threads=4).reshape(-1,4)
    return model,scores,{'iteration':model.best_iteration or rounds,'seconds':time.perf_counter()-start}


def main():
    cache=np.load(CACHE,allow_pickle=False)
    base,y,a,hard=[cache['train_'+k] for k in ('x','y','answers','hard')]
    indices=np.flatnonzero(hard)
    strata=np.asarray([len(set(aa))*10+int(yy.sum()) for aa,yy in zip(a[indices],y[indices])])
    folds=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=7506).split(indices,strata))
    specs=[('lambdarank',7,False),('lambdarank',15,False),('rank_xendcg',15,False),
           ('answer_ce',7,False),('answer_ce',15,False),('answer_ce',15,True)]
    for objective,leaves,math in specs:
        name=f'rank_{objective}_{leaves}'+('_math' if math else '')
        if (OUT/f'cv_{name}.json').exists():
            continue
        features=np.concatenate([base,np.load(OUT/'train_numeric.npy')],axis=2) if math else base
        config={'objective':objective,'leaves':leaves,'numeric':math,'kind':'raw'}
        oof=np.zeros((len(y),4),dtype=np.float32); records=[]
        for fold,(_,valid_i) in enumerate(folds):
            vi=indices[valid_i]; ti=np.setdiff1d(np.arange(len(base)),vi)
            model,scores,info=fit_ranking(features[ti],y[ti],a[ti],features[vi],y[vi],a[vi],config)
            oof[vi]=scores
            pooled,_,_,_=pooled_logits(scores,a[vi]); chosen=pooled.argmax(1)
            info.update(fold=fold,correct=int(y[vi,chosen].sum()),n=len(vi)); records.append(info)
            model.save_model(str(OUT/f'cv_{name}_{fold}.txt'))
            print(f'{name} fold {fold}: {100*info["correct"]/len(vi):.2f}%, trees={info["iteration"]}, {info["seconds"]:.1f}s',flush=True)
        pooled,_,_,_=pooled_logits(oof[indices],a[indices]); chosen=pooled.argmax(1)
        correct=int(y[indices,chosen].sum())
        report={'name':name,'config':config,'correct':correct,'n':len(indices),'oof_accuracy':100*correct/len(indices),
                'folds':records,'median_iteration':int(np.median([r['iteration'] for r in records])),
                'total_training_seconds':sum(r['seconds'] for r in records)}
        np.save(OUT/f'cv_{name}_oof.npy',oof)
        (OUT/f'cv_{name}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('CV '+json.dumps({k:v for k,v in report.items() if k not in ('config','folds')}),flush=True)


if __name__=='__main__':
    main()
