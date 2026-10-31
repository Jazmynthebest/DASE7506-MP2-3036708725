"""Low-dimensional nonnegative stacking fitted to train-only out-of-fold scores."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '4'
import json
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from round2_features import answer_masks
from round2_nn import pooled_logits,ranking_loss_gradient
from train import ROOT,CACHE

OUT = ROOT/'runs'/'round2'


def normalized_scores(scores,answers):
    z,valid,_,_ = pooled_logits(scores,answers)
    mean = (np.where(valid,z,0)).sum(1,keepdims=True)/valid.sum(1,keepdims=True)
    delta = np.where(valid,z-mean,0)
    scale = np.sqrt((delta*delta).sum(1,keepdims=True)/valid.sum(1,keepdims=True)+0.25**2)
    return delta/scale


def family_definitions():
    return {
        'hybrid':['raw_answer_aux'],
        'pair':['pair_raw64_l2_0.001','pair_raw64_l2_0.01'],
        'legacy':['legacy_seed2026','legacy_seed2027','legacy_seed2028'],
        'context':['raw_control','context_64','groups_deep'],
        'tree':['tree_raw_math7','tree_group_math15'],
        'kernel':['kernel_g0.5_r1.0'],
    }


def fit_weights(x,y,answers,regularization):
    def objective(w):
        z = x @ w
        loss,g = ranking_loss_gradient(z,y,answers)
        return loss+0.5*regularization*np.dot(w,w),np.einsum('ncm,nc->m',x,g)+regularization*w
    result = minimize(objective,np.ones(x.shape[-1])/x.shape[-1],jac=True,
                      method='L-BFGS-B',bounds=[(0,10)]*(x.shape[-1]-1)+[(-2,2)],
                      options={'maxiter':150,'ftol':1e-9})
    return result.x.astype(np.float32)


def main():
    cache = np.load(CACHE,allow_pickle=False)
    y,answers,hard = [cache['train_'+k] for k in ('y','answers','hard')]
    inds = np.flatnonzero(hard); a = answers[inds]; y = y[inds]
    families = family_definitions()
    parts = []
    for name,models in families.items():
        part = np.mean([normalized_scores(np.load(OUT/f'cv_{m}_oof.npy')[inds],a) for m in models],axis=0)
        parts.append(part)
    valid,counts,_,_ = answer_masks(a)
    consensus = np.log(np.maximum(counts,1))*valid
    consensus -= (consensus.sum(1,keepdims=True)/valid.sum(1,keepdims=True))*valid
    parts.append(consensus)
    x = np.stack(parts,axis=2)
    strata = np.asarray([len(set(aa))*10+int(yy.sum()) for aa,yy in zip(a,y)])
    folds = list(StratifiedKFold(n_splits=3,shuffle=True,random_state=7507).split(x,strata))
    trials = []
    for reg in (0.001,0.01,0.1):
        scores = np.zeros((len(y),4),dtype=np.float32); fold_weights = []
        for ti,vi in folds:
            w = fit_weights(x[ti],y[ti],a[ti],reg)
            scores[vi] = x[vi] @ w; fold_weights.append(w.tolist())
        z,_,_,_ = pooled_logits(scores,a); choices = z.argmax(1)
        correct = int(y[np.arange(len(y)),choices].sum())
        full_w = fit_weights(x,y,a,reg)
        trials.append({'regularization':reg,'correct':correct,'n':len(y),'meta_cv_accuracy':100*correct/len(y),
                       'weights':full_w.tolist(),'fold_weights':fold_weights})
    trials.sort(key=lambda t:(-t['correct'],-t['regularization']))
    report = {'families':families,'feature_order':list(families)+['consensus'],
              'trials':trials,'selected':trials[0],
              'note':'Development estimate based on train-only base OOF predictions. Validation and test are not read.'}
    (OUT/'stack_selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
