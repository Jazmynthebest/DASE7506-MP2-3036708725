"""Group-consistent neural ranking, trained entirely from scratch with NumPy."""
import time
import numpy as np
from scipy.special import expit,logsumexp
from round2_features import answer_masks


def pooled_logits(logits,answers):
    valid,counts,mean,_ = answer_masks(answers)
    scores = (mean @ logits[:,:,None])[:,:,0]
    return np.where(valid,scores,-1e4),valid,counts,mean


def ranking_loss_gradient(logits,labels,answers,kind='answer'):
    if kind == 'binary':
        valid = answers != 'INVALID'
        loss = ((np.logaddexp(0,logits)-labels*logits)*valid).sum()/valid.sum()
        return float(loss),(expit(logits)-labels)*valid/valid.sum()
    if kind == 'candidate':
        valid,counts,mean,_ = answer_masks(answers)
        z = np.where(valid,logits,-1e4)
        normalizer = logsumexp(z,axis=1,keepdims=True)
        positive = logsumexp(np.where(labels,z,-1e4),axis=1,keepdims=True)
        gradient = np.exp(z-normalizer)-np.where(labels,np.exp(z-positive),0)
        return float((normalizer-positive).mean()),gradient/len(z)
    scores,valid,counts,mean = pooled_logits(logits,answers)
    z = np.where(valid,scores-np.log(np.maximum(counts,1)),-1e4)
    normalizer = logsumexp(z,axis=1,keepdims=True)
    positive = labels/labels.sum(1,keepdims=True)
    loss = float((normalizer[:,0]-(positive*scores).sum(1)).mean())
    gradient = (np.exp(z-normalizer)-positive)/len(z)
    return loss,(mean.transpose(0,2,1) @ gradient[:,:,None])[:,:,0]


def initialize(d,config,rng):
    h = config['hidden']; second = config.get('second',0)
    weights = {'w0':rng.normal(0,np.sqrt(1/d),(d,h)).astype(np.float32),
               'b0':np.zeros(h,dtype=np.float32),'skip':np.zeros(d,dtype=np.float32)}
    if second:
        weights.update({'w1':rng.normal(0,np.sqrt(1/h),(h,second)).astype(np.float32),
                        'b1':np.zeros(second,dtype=np.float32),
                        'out':rng.normal(0,0.03,second).astype(np.float32)})
    else:
        weights['out'] = rng.normal(0,0.03,h).astype(np.float32)
    return weights


def forward(x,weights,config,rng=None):
    pre = x @ weights['w0']+weights['b0']
    h = np.maximum(pre,0)
    masks = []
    drop = config['dropout']
    if rng is not None and drop:
        mask = (rng.random(h.shape) >= drop).astype(np.float32)/(1-drop)
        h *= mask; masks.append(mask)
    else:
        masks.append(1.0)
    if 'w1' in weights:
        pre1 = h @ weights['w1']+weights['b1']
        h1 = np.maximum(pre1,0)
        if rng is not None and drop:
            mask1 = (rng.random(h1.shape) >= drop).astype(np.float32)/(1-drop)
            h1 *= mask1; masks.append(mask1)
        else:
            masks.append(1.0)
        z = h1 @ weights['out']+x @ weights['skip']
        return z,(pre,h,pre1,h1,masks)
    return h @ weights['out']+x @ weights['skip'],(pre,h,masks)


def backward(x,weights,cache,grad,decay):
    grads = {'skip':x.T @ grad+decay*weights['skip']}
    if 'w1' in weights:
        pre,h,pre1,h1,masks = cache
        grads['out'] = h1.T @ grad+decay*weights['out']
        gh1 = grad[:,None]*weights['out']*masks[1]*(pre1>0)
        grads['w1'] = h.T @ gh1+decay*weights['w1']
        grads['b1'] = gh1.sum(0)
        gh = (gh1 @ weights['w1'].T)*masks[0]*(pre>0)
    else:
        pre,h,masks = cache
        grads['out'] = h.T @ grad+decay*weights['out']
        gh = grad[:,None]*weights['out']*masks[0]*(pre>0)
    grads['w0'] = x.T @ gh+decay*weights['w0']
    grads['b0'] = gh.sum(0)
    return grads


def predict(x,weights,config):
    n = len(x)
    return forward(x.reshape(-1,x.shape[-1]),weights,config)[0].reshape(n,4)


def fit(x,y,answers,vx,vy,vanswers,config,seed=2026,fixed_epochs=None,verbose=False):
    start = time.perf_counter()
    mean = x.reshape(-1,x.shape[-1]).mean(0)
    scale = np.maximum(x.reshape(-1,x.shape[-1]).std(0),config.get('scale_floor',0.05))
    x = np.ascontiguousarray((x-mean)/scale,dtype=np.float32)
    vx = np.ascontiguousarray((vx-mean)/scale,dtype=np.float32)
    rng = np.random.default_rng(seed)
    weights = initialize(x.shape[-1],config,rng)
    ema = {k:w.copy() for k,w in weights.items()}
    m = {k:np.zeros_like(w) for k,w in weights.items()}
    v = {k:np.zeros_like(w) for k,w in weights.items()}
    best = None; best_epoch = 0; best_loss = 1e30; best_accuracy = 0; step = 0
    history = []; curve_scores = []
    max_epochs = fixed_epochs or config.get('epochs',120)
    for epoch in range(1,max_epochs+1):
        order = rng.permutation(len(x))
        lr = config.get('lr',0.0015)*max(0.15,1-epoch/150)
        batch_size = config.get('batch_groups',64)
        for start_batch in range(0,len(order),batch_size):
            inds = order[start_batch:start_batch+batch_size]
            batch = x[inds].reshape(-1,x.shape[-1])
            z,cache = forward(batch,weights,config,rng)
            z = z.reshape(-1,4)
            _,grad = ranking_loss_gradient(z,y[inds],answers[inds],config.get('loss','answer'))
            valid = answers[inds] != 'INVALID'
            grad += config.get('auxiliary',0.0)*(expit(z)-y[inds])*valid/max(1,valid.sum())
            grads = backward(batch,weights,cache,grad.ravel(),config['decay'])
            step += 1
            # A global gradient clip limits rare, extreme numeric feature batches.
            norm = np.sqrt(sum(float(np.sum(g*g)) for g in grads.values()))
            factor = min(1.0,5/max(norm,1e-8))
            for key,w in weights.items():
                g = grads[key]*factor
                m[key] *= 0.9; m[key] += 0.1*g
                v[key] *= 0.999; v[key] += 0.001*g*g
                w -= lr*(m[key]/(1-0.9**step))/(np.sqrt(v[key]/(1-0.999**step))+1e-8)
                ema_factor = config.get('ema',0.99)
                ema[key] *= ema_factor; ema[key] += (1-ema_factor)*w
        if epoch % 4 and epoch != max_epochs:
            continue
        scores = predict(vx,ema,config)
        loss,_ = ranking_loss_gradient(scores,vy,vanswers,config.get('loss','answer'))
        pooled,valid,_,_ = pooled_logits(scores,vanswers)
        chosen = pooled.argmax(1)
        acc = float(100*vy[np.arange(len(vy)),chosen].mean())
        history.append({'epoch':epoch,'accuracy':acc,'loss':loss})
        if config.get('curve_epochs'):
            curve_scores.append(scores.astype(np.float32))
        # Continuous held-out ranking loss chooses the epoch, reducing accuracy-maximization noise.
        if loss < best_loss-1e-5 or fixed_epochs:
            best_loss,best_accuracy,best_epoch = loss,acc,epoch
            best = {k:w.copy() for k,w in ema.items()}
        if verbose and epoch % 20 == 0:
            print(f'epoch={epoch} acc={acc:.2f} loss={loss:.4f} best_epoch={best_epoch}',flush=True)
        if fixed_epochs is None and epoch-best_epoch >= 28:
            break
    model = {**best,'mean':mean,'scale':scale}
    scores = predict(vx,best,config)
    if config.get('_trace_path'):
        np.savez(config['_trace_path'],epochs=np.asarray([h['epoch'] for h in history]),
                 scores=np.asarray(curve_scores,dtype=np.float32))
    return model,scores,{'best_epoch':best_epoch,'epochs_run':epoch,'selection_loss':best_loss,
                          'selection_accuracy':best_accuracy,'seconds':time.perf_counter()-start,'history':history}


def predict_saved(x,model,config):
    x = (x-model['mean'])/model['scale']
    weights = {k:model[k] for k in ('w0','b0','out','skip','w1','b1') if k in model}
    return predict(x,weights,config)
