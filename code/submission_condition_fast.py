"""Exact structural-blend inference with one-time tree-array loading per call."""
from pathlib import Path
import json
import numpy as np
from common import parse_answer
from features import make_features
from submission_compact import compact_features,initial_features,answer_groups
from numpy_trees import predict_trees
from structure_math import program_vector,compatibility
from round6_parser import chained_program


def predict_scores(bank,arrays,checkpoint=None):
    if not bank:return np.empty((0,4),dtype=np.float64)
    root=Path(checkpoint) if checkpoint is not None else Path(__file__).resolve().parent/'checkpoints/condition_blend'
    config=json.loads((root/'config.json').read_text())
    answers=np.asarray([['INVALID' if (a:=parse_answer(c['text'])) is None else a for c in row['candidates']] for row in bank])
    base=make_features(bank,arrays,'full')
    with np.load(root/'semantic_probes.npz',allow_pickle=False) as model:semantic=compact_features(base,answers,model)
    with np.load(root/'pair.npz',allow_pickle=False) as model:pair=base@model['weight']
    with np.load(root/'calibration.npz',allow_pickle=False) as model:
        initial=initial_features(pair,semantic,answers)@model['weight']
    with np.load(root/'readout.npz',allow_pickle=False) as model:
        expected=base[:,:,:1024].mean(1)@model['weight']+model['bias']
    actual=[];parsed=[]
    for row in bank:
        programs=[chained_program(row['question'],c['text']) for c in row['candidates']]
        actual.append([program_vector(p) for p in programs]);parsed.append([p is not None for p in programs])
    features=np.concatenate([semantic,compatibility(np.asarray(actual),np.asarray(parsed),expected)],axis=2)
    with np.load(root/'trees.npz',allow_pickle=False) as archive:trees={key:archive[key] for key in archive.files}
    raw=predict_trees(features.reshape(-1,features.shape[-1]),trees).reshape(-1,4)+initial
    weight=config.get('structure_weight',1.)
    if weight!=1.:
        with np.load(root/'compact_trees.npz',allow_pickle=False) as archive:trees={key:archive[key] for key in archive.files}
        compact=predict_trees(semantic.reshape(-1,294),trees).reshape(-1,4)+initial
        raw=weight*raw+(1-weight)*compact
    valid,_,own,_=answer_groups(answers)
    return np.where(valid,(own@raw[:,:,None])[:,:,0],-1e4)


def predict(bank,arrays,checkpoint=None):
    return [int(i) for i in predict_scores(bank,arrays,checkpoint).argmax(1)]
