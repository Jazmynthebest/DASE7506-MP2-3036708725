"""Reproduce the report's post-freeze validation ablations; never fit a model."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
import json
from pathlib import Path
import numpy as np
from common import load_inputs,parse_answer,read_jsonl
from features import make_features,majority_choices
from submission_compact import compact_features,initial_features,answer_groups
from structure_math import program_vector,compatibility
from round6_parser import chained_program
from numpy_trees import predict_trees

ROOT=Path(__file__).resolve().parent

def model(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}

def main():
    bank,arrays=load_inputs('validation')
    refs=read_jsonl(ROOT/'data/gsm8k_validation.jsonl')
    assert [r['id'] for r in refs]==[r['id'] for r in bank]
    answers=np.asarray([[parse_answer(c['text']) or 'INVALID' for c in r['candidates']] for r in bank])
    gold=np.asarray([parse_answer(r['answer']) for r in refs])
    y=answers==gold[:,None]
    hard=np.asarray([len(set(a)-{'INVALID'})>1 for a in answers])
    checkpoint=ROOT/'checkpoints/condition_blend';base=make_features(bank,arrays,'full')
    semantic=compact_features(base,answers,model(checkpoint/'semantic_probes.npz'))
    initial=initial_features(base@model(checkpoint/'pair.npz')['weight'],semantic,answers)@model(checkpoint/'calibration.npz')['weight']
    readout=model(checkpoint/'readout.npz')
    expected=base[:,:,:1024].mean(1)@readout['weight']+readout['bias']
    actual=[];parsed=[]
    for row in bank:
        ps=[chained_program(row['question'],c['text']) for c in row['candidates']]
        actual.append([program_vector(p) for p in ps]);parsed.append([p is not None for p in ps])
    structure=np.concatenate([semantic,compatibility(np.asarray(actual),np.asarray(parsed),expected)],axis=2)
    compact=predict_trees(semantic.reshape(-1,294),model(checkpoint/'compact_trees.npz')).reshape(-1,4)+initial
    arithmetic=predict_trees(structure.reshape(-1,387),model(checkpoint/'trees.npz')).reshape(-1,4)+initial
    raw=.5*compact+.5*arithmetic;valid,_,own,_=answer_groups(answers)
    ablations=[];good=None;scores=None
    for name,z,pool in [('Compact branch only',compact,True),('Arithmetic branch only',arithmetic,True),
                       ('Final blend without answer pooling',raw,False),('Final blend',raw,True)]:
        scores=np.where(valid,(own@z[:,:,None])[:,:,0] if pool else z,-1e4)
        good=y[np.arange(len(y)),scores.argmax(1)]
        ablations.append(dict(method=name,correct=int(good.sum()),n=len(y),accuracy=100*float(good.mean()),
            hard_correct=int(good[hard].sum()),hard_n=int(hard.sum()),hard_accuracy=100*float(good[hard].mean())))
    majority=np.asarray(majority_choices(bank));mg=y[np.arange(len(y)),majority]
    margins=np.zeros(len(y))
    for i in np.flatnonzero(hard):
        unique={a:scores[i,j] for j,a in enumerate(answers[i]) if a!='INVALID'}
        ordered=sorted(unique.values(),reverse=True);margins[i]=ordered[0]-ordered[1]
    ordered=np.flatnonzero(hard)[np.argsort(margins[hard])]
    bins=[dict(quintile=k+1,n=len(ids),correct=int(good[ids].sum()),accuracy=100*float(good[ids].mean()))
          for k,ids in enumerate(np.array_split(ordered,5))]
    result=dict(scope='Fixed-checkpoint validation analysis; no fitting or test access.',ablations=ablations,
        rescued_plurality=int((hard&good&~mg).sum()),introduced_errors=int((hard&~good&mg).sum()),margin_quintiles=bins)
    recorded=json.loads((ROOT.parent/'evidence/report_data.json').read_text())
    assert result['ablations']==recorded['ablations']
    assert result['rescued_plurality']==recorded['error_analysis']['rescued_plurality']
    assert result['introduced_errors']==recorded['error_analysis']['introduced_errors']
    assert [b['correct'] for b in bins]==[b['correct'] for b in recorded['error_analysis']['margin_quintiles']]
    output=ROOT/'runs/validation_analysis.json';output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
