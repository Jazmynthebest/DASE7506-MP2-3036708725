"""Learn question-level arithmetic expectations from supplied training derivations."""
import numpy as np
from sklearn.linear_model import Ridge
from round4_programs import extract_program
from round4_program_verifier import candidate_program

OPERATIONS=['+','-','*','/','^']
CONSTANTS=[2,3,5,7,10,12,60,100]


def program_vector(program):
    vector=np.zeros(22,dtype=np.float32)
    if program is None: return vector
    uses={}; depths=[]
    def visit(node,depth=0):
        kind=node[0]
        depths.append(depth)
        if kind=='q': uses[node[1]]=uses.get(node[1],0)+1
        elif kind=='c':
            for i,value in enumerate(CONSTANTS): vector[14+i]+=abs(node[1]-value)<1e-8
        else:
            if kind in OPERATIONS: vector[OPERATIONS.index(kind)]+=1
            visit(node[1],depth+1); visit(node[2],depth+1)
    visit(program)
    if program[0] in OPERATIONS: vector[5+OPERATIONS.index(program[0])]=1
    vector[10]=len(uses); vector[11]=sum(uses.values()); vector[12]=sum(max(0,v-1) for v in uses.values())
    vector[13]=max(depths,default=0)
    vector[:5]=np.log1p(vector[:5]); vector[10:]=np.log1p(vector[10:])
    return vector


def fit_readout(x,targets,mask,alpha=100.):
    q=x[:,:,:1024].mean(1); q=q[mask]; targets=targets[mask]
    mean=q.mean(0); scale=np.maximum(q.std(0),.03)
    classifier=Ridge(alpha=alpha,solver='cholesky')
    classifier.fit((q-mean)/scale,targets)
    weight=classifier.coef_.T/scale[:,None]
    bias=classifier.intercept_-mean@weight
    return dict(weight=weight.astype(np.float32),bias=bias.astype(np.float32))


def readout(x,model): return x[:,:,:1024].mean(1)@model['weight']+model['bias']


def compatibility(actual,parsed,expected):
    n=len(actual); predicted=np.broadcast_to(expected[:,None,:],actual.shape)
    delta=np.where(parsed[:,:,None],actual-predicted,0)
    dot=(actual*predicted).sum(2)
    denominator=np.maximum(np.sqrt((actual*actual).sum(2)*(predicted*predicted).sum(2)),1e-6)
    summary=np.stack([np.mean(delta*delta,axis=2),np.mean(np.abs(delta),axis=2),np.max(np.abs(delta),axis=2),
                      dot/denominator,parsed.astype(np.float32)],axis=2)
    return np.concatenate([actual,predicted,delta,np.abs(delta),summary],axis=2).astype(np.float32)
