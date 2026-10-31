"""Question-conditioned structural verifier supervised by training derivations."""
import re,json,math
from collections import Counter
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from common import parse_answer
from features import _clean_math
from round2_symbolic import terms
from round4_programs import extract_program


def candidate_program(question,text):
    final=parse_answer(text)
    if final is None: return None
    cleaned=_clean_math(text)
    cleaned=re.sub(r'\\boxed\{([^{}]*)\}',r'\1',cleaned)
    cleaned=cleaned.replace('$',' ').replace('\\,','').replace('\\approx','=')
    equations=[]
    for left,right in re.findall(r'([\d.() +*/\-\n]+)=\s*([-+]?\d+(?:\.\d+)?)',cleaned):
        left=left.strip()
        if len(left)>180 or not re.search(r'[+*/\-]',left): continue
        equations.append('<<'+left+'='+right+'>>')
    return extract_program(question,' '.join(equations)+'\n#### '+final)


def signatures(program):
    if program is None: return []
    result=Counter(); quantities=Counter()
    def visit(node,depth=0):
        kind=node[0]
        if kind=='q':
            quantities[str(node[1])]+=1
            return 'q'+str(node[1])
        if kind=='c':
            result['constant:'+str(node[1])]+=1
            return 'c'+str(node[1])
        result['operation:'+kind]+=1
        result['depth:'+str(min(depth,4))+':'+kind]+=1
        left=visit(node[1],depth+1); right=visit(node[2],depth+1)
        result['children:'+kind+':'+node[1][0]+':'+node[2][0]]+=1
        if node[1][0] in ('q','c'): result['left:'+kind+':'+left]+=1
        if node[2][0] in ('q','c'): result['right:'+kind+':'+right]+=1
        return kind
    visit(program)
    result['root:'+program[0]]=1
    for variable,count in quantities.items():
        result['uses:q'+variable]=1
        if count>1: result['repeats:q'+variable]=min(count-1,3)
    result['distinct_inputs:'+str(len(quantities))]=1
    result['operation_count:'+str(min(12,sum(v for k,v in result.items() if k.startswith('operation:'))))]=1
    return list(result.items())[:45]


def structural_matrix(questions,programs,n_features=131072):
    def rows():
        for question,program in zip(questions,programs):
            structure=signatures(program)
            phrases=terms(question)[:80]
            if not structure:
                yield [('unparsed',1.)]; continue
            normalizer=1/math.sqrt(max(1,len(phrases)*len(structure)))
            features=[('structure:'+key,float(value)*.1) for key,value in structure]
            for phrase in phrases:
                features.extend((phrase+'|'+key,float(value)*normalizer) for key,value in structure)
            yield features
    return FeatureHasher(n_features=n_features,input_type='pair',alternate_sign=True,dtype=np.float32).transform(rows()).tocsr()


def fit_predict(matrix,labels,mask,train_indices,valid_indices,c):
    rows=(5*train_indices[:,None]+np.arange(5)).ravel(); rows=rows[mask[rows]]
    weights=np.where(rows%5==4,.5,1.)
    classifier=LogisticRegression(C=c,solver='liblinear',max_iter=200,random_state=7537)
    classifier.fit(matrix[rows],labels[rows],sample_weight=weights)
    held=(5*valid_indices[:,None]+np.arange(4)).ravel()
    scores=classifier.decision_function(matrix[held]); scores=np.where(mask[held],scores,0.)
    return scores.reshape(-1,4)
