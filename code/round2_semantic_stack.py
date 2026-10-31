"""Nested grouped CV: embedding-only probes followed by a compact contextual ranker."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='4'
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,KFold
from optimize_round2_linear import fit_pairwise
from optimize_round2_ranking import fit_ranking
from round2_features import answer_masks
from round2_nn import pooled_logits
from train import ROOT,CACHE

OUT=ROOT/'runs'/'round2'


def fit_probes(x,y,a):
    d=1024; raw=x[:,:,:d].reshape(-1,d)
    mean=raw.mean(0); scale=np.maximum(raw.std(0),0.03)
    valid,counts,_,_=answer_masks(a)
    w=(valid/np.maximum(counts,1)).ravel()
    fit_rows=np.flatnonzero(valid.ravel())
    classifier=LogisticRegression(C=0.1,solver='lbfgs',max_iter=180,random_state=7506)
    classifier.fit(((raw-mean)/scale)[fit_rows],y.ravel()[fit_rows],sample_weight=w[fit_rows])
    coefficient=classifier.coef_[0]/scale
    bias=classifier.intercept_[0]-np.dot(coefficient,mean)
    pair,_,_=fit_pairwise(x[:,:,:d],y,a,x[:1,:,:d],0.01)
    return {'binary_weight':coefficient.astype(np.float32),'binary_bias':np.asarray(bias,dtype=np.float32),
            'pair_weight':pair['weight']}


def probe_scores(x,model):
    u=x[:,:,:1024]
    return np.stack([u @ model['binary_weight']+model['binary_bias'],u @ model['pair_weight']],axis=2)


def small_features(base,answers,scores):
    valid,counts,same,other=answer_masks(answers)
    features=[base[:,:,2048:]]
    for k in range(scores.shape[-1]):
        z=scores[:,:,k]; mean=z.mean(1,keepdims=True)
        group=(same @ z[:,:,None])[:,:,0]
        rivals=(other @ z[:,:,None])[:,:,0]
        features += [z[:,:,None],(z-mean)[:,:,None],group[:,:,None],rivals[:,:,None],(group-rivals)[:,:,None]]
    return np.concatenate(features,axis=2).astype(np.float32)


def main():
    cache=np.load(CACHE,allow_pickle=False)
    x,y,a,hard=[cache['train_'+k] for k in ('x','y','answers','hard')]
    indices=np.flatnonzero(hard); easy=np.flatnonzero(~hard)
    strata=np.asarray([len(set(aa))*10+int(yy.sum()) for aa,yy in zip(a[indices],y[indices])])
    folds=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=7506).split(indices,strata))
    for fold,(train_i,valid_i) in enumerate(folds):
        path=OUT/f'semantic_nested_features_{fold}.npz'
        if path.exists():
            continue
        ti,vi=indices[train_i],indices[valid_i]
        scores=np.zeros((len(ti),4,2),dtype=np.float32)
        for inner_train,inner_val in KFold(n_splits=3,shuffle=True,random_state=7509).split(ti):
            fitting=np.r_[ti[inner_train],easy]
            probes=fit_probes(x[fitting],y[fitting],a[fitting])
            scores[inner_val]=probe_scores(x[ti[inner_val]],probes)
        outer_fit=np.r_[ti,easy]
        probes=fit_probes(x[outer_fit],y[outer_fit],a[outer_fit])
        validation_scores=probe_scores(x[vi],probes)
        np.savez(path,train_features=small_features(x[ti],a[ti],scores),
                 validation_features=small_features(x[vi],a[vi],validation_scores),train_indices=ti,validation_indices=vi)
        print('Nested probe features prepared, fold '+str(fold),flush=True)
    for objective in ('lambdarank','answer_ce'):
        for leaves in (7,15):
            name=f'semantic_{objective}_{leaves}'
            if (OUT/f'cv_{name}.json').exists():
                continue
            oof=np.zeros((len(x),4),dtype=np.float32); records=[]
            config={'objective':objective,'leaves':leaves,'l2':10}
            for fold in range(3):
                data=np.load(OUT/f'semantic_nested_features_{fold}.npz',allow_pickle=False)
                ti,vi=data['train_indices'],data['validation_indices']
                model,scores,info=fit_ranking(data['train_features'],y[ti],a[ti],data['validation_features'],y[vi],a[vi],config)
                oof[vi]=scores
                z,_,_,_=pooled_logits(scores,a[vi]); choices=z.argmax(1)
                info.update(fold=fold,correct=int(y[vi,choices].sum()),n=len(vi)); records.append(info)
                print(f'{name} fold {fold}: {100*info["correct"]/len(vi):.2f}%, {info["iteration"]} trees',flush=True)
            z,_,_,_=pooled_logits(oof[indices],a[indices]); choices=z.argmax(1); correct=int(y[indices,choices].sum())
            report={'name':name,'config':config,'correct':correct,'n':len(indices),'oof_accuracy':100*correct/len(indices),
                    'folds':records,'median_iteration':int(np.median([r['iteration'] for r in records])),
                    'total_training_seconds':sum(r['seconds'] for r in records),
                    'protocol':'Outer and inner folds grouped by question; embedding probes for ranker training are inner OOF.'}
            np.save(OUT/f'cv_{name}_oof.npy',oof)
            (OUT/f'cv_{name}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print('CV '+json.dumps({k:v for k,v in report.items() if k not in ('config','folds')}),flush=True)


if __name__=='__main__':
    main()
