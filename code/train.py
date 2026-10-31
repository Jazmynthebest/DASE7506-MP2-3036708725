"""Reproducible train/validation-only experiments; test data cannot be loaded here."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '4'

import numpy as np
from common import DATA, load_inputs, parse_answer, read_jsonl
from features import answers_for, choose_scores, majority_choices, make_features

ROOT = Path(__file__).resolve().parent
RUNS = ROOT/'runs'
CACHE = RUNS/'development_cache.npz'


def labels_for(bank, split):
    if split not in ('train', 'validation'):
        raise ValueError('Training script may only access train and validation references')
    refs = read_jsonl(DATA/f'gsm8k_{split}.jsonl')
    labels = []
    for row, ref in zip(bank, refs):
        assert row['id'] == ref['id'] and row['question'] == ref['question']
        answer = parse_answer(ref['answer'])
        labels.append([int(a is not None and a == answer) for a in answers_for([row])[0]])
    y = np.asarray(labels, dtype=np.float32)
    assert y.shape == (len(bank),4) and np.all(y.sum(1) >= 1)
    return y


def accuracy(y, choices, mask=None):
    correct = y[np.arange(len(y)), choices]
    return float(100*correct.mean() if mask is None else 100*correct[mask].mean())


def prepare():
    started = time.perf_counter()
    RUNS.mkdir(exist_ok=True)
    out, stats = {}, {}
    for split in ('train','validation'):
        bank, arrays = load_inputs(split)
        y = labels_for(bank, split)
        x = make_features(bank, arrays, 'full')
        answers = answers_for(bank)
        hard = np.asarray([len({a for a in aa if a is not None}) >= 2 for aa in answers])
        out[f'{split}_x'] = x
        out[f'{split}_y'] = y
        out[f'{split}_hard'] = hard
        # Groups are used for label-free pooling and never become inference assets.
        out[f'{split}_answers'] = np.asarray([['INVALID' if a is None else a for a in aa] for aa in answers])
        majority = majority_choices(bank)
        stats[split] = {
            'questions': len(bank), 'hard_questions': int(hard.sum()),
            'correct_candidate_counts': dict(Counter(int(n) for n in y.sum(1))),
            'first_accuracy': accuracy(y, np.zeros(len(y), dtype=int)),
            'majority_accuracy': accuracy(y, majority),
            'majority_hard_accuracy': accuracy(y, majority, hard),
            'uniform_random_expected_accuracy': float(100*y.mean()),
            'features': x.shape[-1],
        }
    np.savez(CACHE, **out)
    stats['preparation_seconds'] = time.perf_counter()-started
    stats['hardware'] = {'platform': platform.platform(), 'cpu': os.environ.get('PROCESSOR_IDENTIFIER'),
                         'python': platform.python_version(), 'numpy': np.__version__, 'threads': 4}
    (RUNS/'data_summary.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps(stats, indent=2), flush=True)


def feature_columns(mode, n):
    if mode == 'embedding':
        return np.arange(1024)
    if mode == 'relative':
        return np.arange(2048)
    if mode == 'text':
        return np.arange(2048, n-4)
    if mode == 'full':
        return np.arange(n)
    raise ValueError(mode)


def choice_from_cached(scores, answers, pooling):
    scores = np.asarray(scores).reshape(-1,4).copy()
    for i, aa in enumerate(answers):
        for answer in set(aa):
            js = np.flatnonzero(aa == answer)
            if answer == 'INVALID':
                scores[i,js] = -1e20
            elif pooling:
                z = scores[i,js].copy()
                scores[i,js] = (1-pooling)*z+pooling*z.mean()
    return scores.argmax(1)


def record(name, scores, train_seconds, details, assets):
    with np.load(CACHE, allow_pickle=False) as cache:
        y, hard, answers = [cache['validation_'+k] for k in ('y','hard','answers')]
    reports = []
    for pooling in (0.0, 0.5, 1.0):
        choices = choice_from_cached(scores, answers, pooling)
        reports.append({'pooling': pooling, 'accuracy': accuracy(y, choices),
                        'hard_accuracy': accuracy(y, choices, hard),
                        'correct': int(y[np.arange(len(y)),choices].sum())})
    # Hard validation mirrors the declared test/audit construction; no test is consulted.
    best = max(reports, key=lambda r: (r['hard_accuracy'],r['accuracy'],-r['pooling']))
    report = {'name': name, **best, 'pooling_results': reports, 'training_seconds': train_seconds,
              'details': details, 'asset_bytes': sum(p.stat().st_size for p in assets)}
    with (RUNS/'experiments.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(report)+'\n')
    np.save(RUNS/f'{name}-validation-logits.npy', np.asarray(scores, dtype=np.float32).reshape(-1,4))
    (RUNS/f'{name}.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('details','pooling_results')}), flush=True)
    return report


def train_linear():
    from scipy.optimize import minimize
    from scipy.special import expit, logsumexp
    cache = np.load(CACHE, allow_pickle=False)
    full = cache['train_x']; vfull = cache['validation_x']
    y = cache['train_y']; n = len(y)
    for mode in ('embedding','text','full'):
        cols = feature_columns(mode, full.shape[-1])
        raw = full[:,:,cols].reshape(-1,len(cols))
        mean = raw.mean(0)
        scale = np.maximum(raw.std(0),0.03)
        x = np.ascontiguousarray((raw-mean)/scale, dtype=np.float64)
        vx = np.ascontiguousarray((vfull[:,:,cols].reshape(-1,len(cols))-mean)/scale)
        for loss in ('binary','ranking'):
            for regularization in (0.001, 0.03, 0.3):
                tick = time.perf_counter()
                lam = regularization
                if loss == 'ranking':
                    informative = y.min(1) == 0
                    xx = x.reshape(n,4,-1)[informative].reshape(-1,len(cols))
                    yy = y[informative].astype(bool)
                    def objective(w):
                        z = (xx @ w).reshape(-1,4)
                        all_lse = logsumexp(z,axis=1,keepdims=True)
                        pos_lse = logsumexp(np.where(yy,z,-1e30),axis=1,keepdims=True)
                        loss_value = (all_lse-pos_lse).mean()+0.5*lam*np.dot(w,w)
                        grad_z = (np.exp(z-all_lse)-np.where(yy,np.exp(z-pos_lse),0))/len(yy)
                        return loss_value, xx.T @ grad_z.ravel()+lam*w
                else:
                    yy = y.ravel()
                    def objective(w):
                        z = x @ w[:-1]+w[-1]
                        error = (expit(z)-yy)/len(yy)
                        value = np.mean(np.logaddexp(0,z)-yy*z)+0.5*lam*np.dot(w[:-1],w[:-1])
                        return value, np.r_[x.T @ error+lam*w[:-1],error.sum()]
                dim = len(cols)+(loss == 'binary')
                opt = minimize(objective,np.zeros(dim),jac=True,method='L-BFGS-B',
                               options={'maxiter':180,'ftol':1e-8,'gtol':1e-6,'maxcor':10})
                w = opt.x[:-1] if loss == 'binary' else opt.x
                bias = opt.x[-1] if loss == 'binary' else 0.0
                scores = (vx@w+bias).reshape(-1,4)
                name = f'linear_{mode}_{loss}_{regularization:g}'
                path = RUNS/f'{name}.npz'
                # Fold scaling into weights; all remaining arrays are inference assets.
                np.savez(path, columns=cols.astype(np.int32), weight=(w/scale).astype(np.float32),
                         bias=np.asarray(bias-np.dot(mean,w/scale),dtype=np.float32))
                record(name,scores,time.perf_counter()-tick,
                       {'mode':mode,'loss':loss,'l2':lam,'iterations':int(opt.nit),
                        'optimizer_success':bool(opt.success),'seed':0},[path])


def train_trees():
    import lightgbm as lgb
    cache = np.load(CACHE, allow_pickle=False)
    full, vfull = cache['train_x'], cache['validation_x']
    y, vy = cache['train_y'], cache['validation_y']
    hard = cache['validation_hard']
    for mode in ('text','embedding','full'):
        cols = feature_columns(mode,full.shape[-1])
        x = full[:,:,cols].reshape(-1,len(cols)); vx = vfull[:,:,cols].reshape(-1,len(cols))
        for objective in ('binary','lambdarank'):
            for leaves, min_leaf in ((7,80),(15,60)):
                tick = time.perf_counter()
                params = {'objective':objective,'metric':'None','learning_rate':0.035,
                          'num_leaves':leaves,'min_data_in_leaf':min_leaf,'lambda_l2':5.0,
                          'feature_fraction':0.8,'bagging_fraction':0.85,'bagging_freq':1,
                          'num_threads':4,'verbosity':-1,'seed':2026,'deterministic':True,
                          'force_col_wise':True,'max_bin':63}
                if objective == 'lambdarank':
                    params['lambdarank_truncation_level'] = 4
                train = lgb.Dataset(x,label=y.ravel(),group=[4]*len(y) if objective != 'binary' else None)
                valid = lgb.Dataset(vx,label=vy.ravel(),reference=train,
                                    group=[4]*len(vy) if objective != 'binary' else None)
                def evaluate(pred,dataset):
                    chosen = pred.reshape(-1,4).argmax(1)
                    return 'hard_selection_accuracy',accuracy(vy,chosen,hard),True
                model = lgb.train(params,train,num_boost_round=650,valid_sets=[valid],feval=evaluate,
                                  callbacks=[lgb.early_stopping(100,verbose=False)])
                scores = model.predict(vx,raw_score=True,num_threads=4).reshape(-1,4)
                name = f'tree_{mode}_{objective}_{leaves}'
                path = RUNS/f'{name}.txt'
                model.save_model(str(path))
                meta = RUNS/f'{name}-columns.npy'; np.save(meta,cols.astype(np.int32))
                record(name,scores,time.perf_counter()-tick,
                       {'mode':mode,'objective':objective,'leaves':leaves,'min_leaf':min_leaf,
                        'best_iteration':model.best_iteration,'params':params,'seed':2026},[path,meta])


def train_neural(loss_kind='set'):
    """Small residual MLP, trained from scratch with NumPy/Adam and set-valued labels."""
    from scipy.special import expit, logsumexp
    cache = np.load(CACHE, allow_pickle=False)
    full, vfull = cache['train_x'], cache['validation_x']
    labels, vy, hard = cache['train_y'],cache['validation_y'],cache['validation_hard']
    informative = labels.min(1) == 0
    configs = [('full',64,0.003,2026),('full',64,0.003,2027),('full',64,0.003,2028),
               ('embedding',64,0.003,2026),('full',128,0.01,2026)]
    if loss_kind == 'pairwise':
        configs = [('full',64,0.003,s) for s in (2026,2027,2028)]
    for mode, hidden, decay, seed in configs:
        tick = time.perf_counter()
        rng = np.random.default_rng(seed)
        cols = feature_columns(mode,full.shape[-1])
        raw = full[:,:,cols]
        mean = raw.reshape(-1,len(cols)).mean(0)
        scale = np.maximum(raw.reshape(-1,len(cols)).std(0),0.03)
        x = np.ascontiguousarray((raw-mean)/scale,dtype=np.float32)[informative]
        y = labels[informative].astype(bool)
        vx = np.ascontiguousarray((vfull[:,:,cols]-mean)/scale,dtype=np.float32).reshape(-1,len(cols))
        d = len(cols)
        weights = [rng.normal(0,np.sqrt(1/d),(d,hidden)).astype(np.float32),
                   np.zeros(hidden,dtype=np.float32),
                   rng.normal(0,0.03,(hidden,)).astype(np.float32),
                   np.zeros(d,dtype=np.float32)]
        m = [np.zeros_like(w) for w in weights]; v = [np.zeros_like(w) for w in weights]
        step = 0; best_acc = -1; best_loss = 1e30; best_epoch = 0; best = None
        for epoch in range(1,81):
            order = rng.permutation(len(x))
            learning_rate = 0.0015 * max(0.15,1-epoch/100)
            for start in range(0,len(x),96):
                inds = order[start:start+96]
                batch = x[inds].reshape(-1,d)
                target = y[inds]
                pre = batch @ weights[0]+weights[1]
                hidden_values = np.maximum(pre,0)
                mask = (rng.random(hidden_values.shape) >= 0.2).astype(np.float32)/0.8
                dropped = hidden_values*mask
                z = (dropped @ weights[2]+batch @ weights[3]).reshape(-1,4)
                lse = logsumexp(z,axis=1,keepdims=True)
                pos_lse = logsumexp(np.where(target,z,-1e20),axis=1,keepdims=True)
                if loss_kind == 'pairwise':
                    pair_mask = target[:,:,None] & ~target[:,None,:]
                    pair_grad = -expit(-(z[:,:,None]-z[:,None,:]))*pair_mask
                    pair_grad /= pair_mask.sum(axis=(1,2),keepdims=True)*len(inds)
                    grad = pair_grad.sum(2)-pair_grad.sum(1)
                else:
                    grad = (np.exp(z-lse)-np.where(target,np.exp(z-pos_lse),0))/len(inds)
                # Small auxiliary binary term discourages arbitrary group offsets.
                grad += 0.05*(expit(z)-target)/(len(inds)*4)
                gz = grad.ravel()
                gh = gz[:,None]*weights[2][None,:]*mask*(pre > 0)
                grads = [batch.T @ gh+decay*weights[0],gh.sum(0),
                         dropped.T @ gz+decay*weights[2],batch.T @ gz+decay*weights[3]]
                step += 1
                for k,g in enumerate(grads):
                    m[k] *= 0.9; m[k] += 0.1*g
                    v[k] *= 0.999; v[k] += 0.001*g*g
                    weights[k] -= learning_rate*(m[k]/(1-0.9**step))/(np.sqrt(v[k]/(1-0.999**step))+1e-8)
            if epoch % 2:
                continue
            scores = (np.maximum(vx @ weights[0]+weights[1],0) @ weights[2]+vx @ weights[3]).reshape(-1,4)
            acc = accuracy(vy,scores.argmax(1),hard)
            val_loss = float((logsumexp(scores,axis=1)-logsumexp(np.where(vy,scores,-1e20),axis=1))[hard].mean())
            if acc > best_acc or (acc == best_acc and val_loss < best_loss):
                best_acc, best_loss, best_epoch = acc,val_loss,epoch
                best = [w.copy() for w in weights]
            if epoch % 10 == 0:
                print(f'MLP {mode} h={hidden} seed={seed}: epoch {epoch}, hard={acc:.2f}, best={best_acc:.2f}',flush=True)
            if epoch-best_epoch >= 18:
                break
        w0,b0,w1,skip = best
        scores = (np.maximum(vx @ w0+b0,0) @ w1+vx @ skip).reshape(-1,4)
        name = f'neural_{"pairwise_" if loss_kind == "pairwise" else ""}{mode}_{hidden}_{seed}'
        path = RUNS/f'{name}.npz'
        np.savez(path,columns=cols.astype(np.int32),mean=mean,scale=scale,
                 w0=w0,b0=b0,w1=w1,skip=skip)
        record(name,scores,time.perf_counter()-tick,
               {'mode':mode,'hidden':hidden,'weight_decay':decay,'seed':seed,
                'epochs':epoch,'best_epoch':best_epoch,'learning_rate':0.0015,'dropout':0.2,
                'batch_groups':96,'loss':loss_kind+' ranking + 0.05 binary'},[path])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('prepare','linear','trees','neural','pairwise','all'),default='all')
    args = parser.parse_args()
    if args.stage in ('prepare','all'):
        prepare()
    if args.stage in ('linear','all'):
        train_linear()
    if args.stage in ('trees','all'):
        train_trees()
    if args.stage in ('neural','all'):
        train_neural()
    if args.stage in ('pairwise','all'):
        train_neural('pairwise')


if __name__ == '__main__':
    main()
