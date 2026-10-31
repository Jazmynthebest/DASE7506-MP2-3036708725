"""Convex pairwise ranking with within-question scaling and optional explicit interactions."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '4'
import json
import argparse
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from round2_features import fit_transformer,transform,answer_masks
from round2_nn import pooled_logits
from train import ROOT,CACHE

OUT = ROOT/'runs'/'round2'


def fit_pairwise(tx,labels,answers,vx,regularization=0.01,penalty='l2'):
    valid = answers != 'INVALID'
    positive = labels.astype(bool)
    pairs = positive[:,:,None] & ~positive[:,None,:] & valid[:,None,:]
    q,i,j = np.where(pairs)
    differences = tx[q,i]-tx[q,j]
    scale = np.maximum(differences.std(0),0.03)
    x = np.ascontiguousarray(differences/scale,dtype=np.float64)
    counts = pairs.sum((1,2))
    sample = 1/np.maximum(counts[q],1)
    sample /= sample.sum()
    if penalty == 'l1':
        from sklearn.linear_model import LogisticRegression
        c = regularization
        xx = np.r_[x,-x]; yy = np.r_[np.ones(len(x)),np.zeros(len(x))]
        ww = np.r_[sample,sample]*len(x)
        model = LogisticRegression(solver='liblinear',penalty='l1',fit_intercept=False,
                                   C=c,max_iter=500,random_state=7506,tol=1e-5)
        model.fit(xx,yy,sample_weight=ww)
        weight = model.coef_[0]
        iterations = int(model.n_iter_[0])
    else:
        def objective(w):
            z = x @ w
            return float(np.sum(sample*np.logaddexp(0,-z))+0.5*regularization*np.dot(w,w)),x.T @ (-sample*expit(-z))+regularization*w
        result = minimize(objective,np.zeros(x.shape[-1]),jac=True,method='L-BFGS-B',
                          options={'maxiter':220,'ftol':1e-8,'gtol':1e-6,'maxcor':10})
        weight = result.x; iterations = int(result.nit)
    weight = (weight/scale).astype(np.float32)
    scores = (vx @ weight).astype(np.float32)
    return {'weight':weight},scores,{'iterations':iterations,'nonzero':int((weight != 0).sum())}


def configurations():
    specs = []
    for kind,components in [('raw',64),('context',64),('bilinear',32),('bilinear',64),('quadratic',32)]:
        for reg in (0.001,0.01,0.1):
            specs.append((kind,components,reg,'l2',False))
    for c in (0.003,0.03,0.3):
        specs.append(('raw',64,c,'l1',True))
    return specs


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--proof',action='store_true')
    args = parser.parse_args()
    cache = np.load(CACHE,allow_pickle=False)
    base,labels,answers,hard = [cache['train_'+k] for k in ('x','y','answers','hard')]
    indices = np.flatnonzero(hard)
    strata = np.asarray([len(set(answers[i]))*10+int(labels[i].sum()) for i in indices])
    folds = list(StratifiedKFold(n_splits=3,shuffle=True,random_state=7506).split(indices,strata))
    specs = [('raw',64,reg,'l2',False) for reg in (0.001,0.01,0.1)] if args.proof else configurations()
    for kind,components,reg,penalty,numeric in specs:
        name = f'pair_{kind}{components}_{penalty}_{reg}'+('_proof' if args.proof else '_math' if numeric else '')
        if (OUT/f'cv_{name}.json').exists():
            continue
        extra = np.load(OUT/'train_proof.npy') if args.proof else np.load(OUT/'train_numeric.npy') if numeric else None
        oof = np.zeros((len(base),4),dtype=np.float32); records = []
        for fold,(train_i,valid_i) in enumerate(folds):
            start = time.perf_counter()
            ti,vi = indices[train_i],indices[valid_i]
            path = OUT/f'cv_transform_{kind}_{components}_{fold}.npz'
            if path.exists():
                transformer = dict(np.load(path,allow_pickle=False))
            else:
                transformer = fit_transformer(base[ti],kind,components)
                np.savez(path,**transformer)
            tx = transform(base[ti],answers[ti],transformer,extra[ti] if extra is not None else None)
            vx = transform(base[vi],answers[vi],transformer,extra[vi] if extra is not None else None)
            model,scores,detail = fit_pairwise(tx,labels[ti],answers[ti],vx,reg,penalty)
            oof[vi] = scores
            pooled,_,_,_ = pooled_logits(scores,answers[vi]); chosen = pooled.argmax(1)
            record = {'fold':fold,'correct':int(labels[vi,chosen].sum()),'n':len(vi),'seconds':time.perf_counter()-start,**detail}
            records.append(record)
            print(f'{name} fold {fold}: {100*record["correct"]/len(vi):.2f}%, {record["seconds"]:.1f}s',flush=True)
        pooled,_,_,_ = pooled_logits(oof[indices],answers[indices]); chosen = pooled.argmax(1)
        correct = int(labels[indices,chosen].sum())
        report = {'name':name,'config':{'kind':kind,'components':components,'regularization':reg,'penalty':penalty,'numeric':numeric},
                  'proof':args.proof,'correct':correct,'n':len(indices),'oof_accuracy':100*correct/len(indices),
                  'folds':records,'total_training_seconds':sum(r['seconds'] for r in records)}
        np.save(OUT/f'cv_{name}_oof.npy',oof)
        (OUT/f'cv_{name}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('CV '+json.dumps({k:v for k,v in report.items() if k not in ('config','folds')}),flush=True)


if __name__ == '__main__':
    main()
