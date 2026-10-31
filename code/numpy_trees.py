"""Inference for numeric decision trees exported from a frozen LightGBM model."""
import numpy as np


def predict_trees(features,model):
    # Compare in float64: Python scalar promotion on float32 arrays can round a split.
    x=np.asarray(features,dtype=np.float64)
    if not np.isfinite(x).all(): raise ValueError('Expected finite extracted features')
    scores=np.zeros(len(x),dtype=np.float64)
    for tree in range(len(model['root'])):
        position=np.full(len(x),model['root'][tree],dtype=np.int32)
        active=position>=0
        while active.any():
            rows=np.flatnonzero(active); node=position[rows]
            column=model['feature'][tree,node]
            left=x[rows,column]<=model['threshold'][tree,node]
            position[rows]=np.where(left,model['left'][tree,node],model['right'][tree,node])
            active=position>=0
        scores+=model['value'][tree,-position-1]
    return scores
