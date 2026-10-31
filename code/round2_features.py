"""Second-round, label-free comparison and numeric relation features."""
from fractions import Fraction
import itertools
import math
import re
import numpy as np
from common import parse_answer


SMALL = dict(zip(('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split(),range(20)))
TENS = dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(),range(20,100,10)))
NUMBER = re.compile(r'(?<![\w.])[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:/\d+)?')


def question_numbers(text):
    text = text.lower()
    values = []
    for match in NUMBER.finditer(text):
        try:
            v = float(Fraction(match.group().replace(',','')))
            if abs(v) <= 1e9 and math.isfinite(v):
                values.append(v)
        except (ValueError,ZeroDivisionError):
            pass
    words = re.findall(r'[a-z]+',text)
    for i,w in enumerate(words):
        if w in SMALL:
            if i and words[i-1] in TENS:
                continue
            v = SMALL[w]
        elif w in TENS:
            v = TENS[w]+(SMALL.get(words[i+1],0) if i+1 < len(words) else 0)
        else:
            continue
        if i+1 < len(words) and words[i+1] in ('hundred','thousand'):
            v *= 100 if words[i+1] == 'hundred' else 1000
        values.append(float(v))
    return values[:14]


def _clean_values(values):
    a = np.asarray(values,dtype=np.float64).ravel()
    return a[np.isfinite(a) & (np.abs(a) <= 1e12)]


def relation_sets(values):
    """Reusable arithmetic evidence. Operands must come from distinct question positions."""
    q = np.asarray(values,dtype=np.float64)
    pools = [q, np.asarray([q.sum(),q.prod() if len(q) else 0])]
    if len(q) < 2:
        return pools+[np.empty(0)]*16
    pair_index = np.asarray(list(itertools.permutations(range(len(q)),2)))
    a,b = q[pair_index[:,0]],q[pair_index[:,1]]
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        pools += [a+b,a-b,a*b,a/b,a*b/100,a/(b/100),a*(1+b/100),a*(1-b/100)]
        if len(q) >= 3:
            tri = np.asarray(list(itertools.permutations(range(len(q)),3)))
            a,b,c = q[tri[:,0]],q[tri[:,1]],q[tri[:,2]]
            pools += [(a+b)*c,(a-b)*c,(a+b)/c,(a-b)/c,a*b+c,a*b-c,a/(b+c),a/(b-c)]
        else:
            pools += [np.empty(0)]*8
    return [_clean_values(v) for v in pools]


def numeric_features(bank):
    output = []
    for row in bank:
        question = row['question'].lower()
        numbers = question_numbers(question)
        pools = relation_sets(numbers)
        rows = []
        for candidate in row['candidates']:
            answer = parse_answer(candidate['text'])
            value = float(Fraction(answer)) if answer is not None else 0.0
            value = float(np.clip(value,-1e12,1e12))
            feats = []
            # Fixed scale factors cover common percentage and unit conversions.
            for pool in pools:
                for factor in (1.0,0.5,2.0,60.0,1/60.0,100.0,0.01):
                    if len(pool):
                        error = np.abs(pool*factor-value)/np.maximum(1.0,np.maximum(np.abs(pool*factor),abs(value)))
                        best = float(error.min())
                        feats += [best,float(best < 1e-6)]
                    else:
                        feats += [1.0,0.0]
            feats += [len(numbers),float('minute' in question),float('hour' in question),
                      float('%' in question or 'percent' in question),float('half' in question),
                      float('twice' in question or 'double' in question),
                      float('total' in question),float('remaining' in question or 'left' in question),
                      float(any(abs(value-q)<1e-8 for q in numbers))]
            rows.append(feats)
        a = np.asarray(rows,dtype=np.float32)
        output.append(np.concatenate([a,a-a.mean(0,keepdims=True)],axis=1))
    return np.asarray(output,dtype=np.float32)


def answer_masks(answers):
    answers = np.asarray(answers)
    valid = answers != 'INVALID'
    same = (answers[:,:,None] == answers[:,None,:]) & valid[:,:,None] & valid[:,None,:]
    counts = same.sum(2).astype(np.float32)
    mean = same / np.maximum(counts[:,:,None],1)
    other = valid[:,None,:] & ~same
    other_mean = other/np.maximum(other.sum(2,keepdims=True),1)
    return valid,counts,mean.astype(np.float32),other_mean.astype(np.float32)


def fit_transformer(base,kind='context',n_components=128):
    from sklearn.decomposition import PCA
    unit = base[:,:,:1024].reshape(-1,1024)
    pca = PCA(n_components=n_components,svd_solver='randomized',random_state=7506).fit(unit)
    matrix = pca.components_.T/np.maximum(np.sqrt(pca.explained_variance_),1e-4)
    return {'pca_mean':pca.mean_.astype(np.float32),'pca_matrix':matrix.astype(np.float32),
            'kind':np.asarray(kind),'n_components':np.asarray(n_components)}


def transform(base,answers,transformer,extra=None):
    valid,counts,same,other = answer_masks(answers)
    kind = str(transformer['kind'])
    if kind == 'raw':
        x = base
    else:
        z = (base[:,:,:1024]-transformer['pca_mean']) @ transformer['pca_matrix']
        mean = z.mean(1,keepdims=True)
        delta = z-mean
        own = same @ z
        rival = other @ z
        text = base[:,:,2048:]
        if kind == 'pca':
            x = np.concatenate([z,delta,text],axis=2)
        elif kind == 'relative':
            x = np.concatenate([delta,delta*delta,text],axis=2)
        elif kind == 'kernel':
            x = np.concatenate([delta,text[:,:,:140],base[:,:,-4:]],axis=2)
        elif kind == 'bilinear':
            cross = (delta[:,:,:,None]*mean[:,:,None,:]).reshape(len(base),4,-1)
            x = np.concatenate([z,delta,cross,text],axis=2)
        elif kind == 'quadratic':
            cross = (delta[:,:,:,None]*delta[:,:,None,:]).reshape(len(base),4,-1)
            x = np.concatenate([z,delta,cross,text],axis=2)
        elif kind == 'context':
            x = np.concatenate([z,delta,z*mean,delta*mean,own,rival,own-rival,text],axis=2)
        elif kind == 'groups':
            std = np.sqrt(np.maximum(same @ (z*z)-own*own,0))
            x = np.concatenate([own,rival,own-rival,own*mean,std,
                                same @ text,other @ text],axis=2)
        else:
            raise ValueError(kind)
    if extra is not None:
        x = np.concatenate([x,extra],axis=2)
    return np.ascontiguousarray(x,dtype=np.float32)
