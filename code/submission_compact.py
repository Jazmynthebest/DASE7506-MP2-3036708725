"""Offline compact answer verifier: cross-fitted linear scores + residual trees."""
import json
from pathlib import Path
import numpy as np
from common import parse_answer
from features import make_features


def answer_groups(answers):
    valid=answers!='INVALID'
    same=(answers[:,:,None]==answers[:,None,:])&valid[:,:,None]&valid[:,None,:]
    counts=same.sum(2).astype(np.float32)
    own=same/np.maximum(counts[:,:,None],1)
    other=valid[:,None,:]&~same
    other=other/np.maximum(other.sum(2,keepdims=True),1)
    return valid,counts,own.astype(np.float32),other.astype(np.float32)


def normalize(scores,answers):
    valid,_,own,_=answer_groups(answers)
    group=(own@np.asarray(scores,dtype=np.float32)[:,:,None])[:,:,0]
    mean=np.where(valid,group,0).sum(1,keepdims=True)/valid.sum(1,keepdims=True)
    delta=np.where(valid,group-mean,0)
    return delta/np.sqrt((delta*delta).sum(1,keepdims=True)/valid.sum(1,keepdims=True)+1.)


def compact_features(base,answers,probes):
    unit=base[:,:,:1024]
    z=np.stack([unit@probes['binary_weight']+probes['binary_bias'],unit@probes['pair_weight']],axis=2)
    _,_,own,other=answer_groups(answers)
    features=[base[:,:,2048:]]
    for k in range(2):
        score=z[:,:,k]; group=(own@score[:,:,None])[:,:,0]; rival=(other@score[:,:,None])[:,:,0]
        features.extend([score[:,:,None],(score-score.mean(1,keepdims=True))[:,:,None],
                         group[:,:,None],rival[:,:,None],(group-rival)[:,:,None]])
    return np.concatenate(features,axis=2).astype(np.float32)


def initial_features(pair,semantic,answers):
    valid,counts,_,_=answer_groups(answers)
    consensus=np.log(np.maximum(counts,1))*valid
    consensus-=(consensus.sum(1,keepdims=True)/valid.sum(1,keepdims=True))*valid
    return np.stack([normalize(pair,answers),normalize(semantic[:,:,284],answers),
                     normalize(semantic[:,:,289],answers),consensus],axis=2)


def predict_scores(bank,arrays,checkpoint):
    if not bank: return np.empty((0,4),dtype=np.float32)
    root=Path(checkpoint) if checkpoint is not None else Path(__file__).resolve().parent/'checkpoints'/'compact'
    config=json.loads((root/'config.json').read_text(encoding='utf-8'))
    answers=np.asarray([['INVALID' if (a:=parse_answer(c['text'])) is None else a for c in row['candidates']] for row in bank])
    base=make_features(bank,arrays,'full')
    with np.load(root/'semantic_probes.npz',allow_pickle=False) as probes:
        semantic=compact_features(base,answers,probes)
    with np.load(root/'pair.npz',allow_pickle=False) as model:
        pair=base@model['weight']
    with np.load(root/'calibration.npz',allow_pickle=False) as calibration:
        initial=initial_features(pair,semantic,answers)@calibration['weight']
    import lightgbm as lgb
    model=lgb.Booster(model_file=str(root/'residual.txt'))
    raw=model.predict(semantic.reshape(-1,294),raw_score=True,num_iteration=config['rounds'],num_threads=4).reshape(-1,4)
    valid,_,own,_=answer_groups(answers)
    grouped=(own@(raw+initial)[:,:,None])[:,:,0]
    return np.where(valid,grouped,-1e4)


def predict(bank,arrays,checkpoint):
    return [int(i) for i in predict_scores(bank,arrays,checkpoint).argmax(1)]
