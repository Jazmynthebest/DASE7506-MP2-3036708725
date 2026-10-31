"""Question-resampled complete pipeline; unique-question inner cross-fitting."""
import time,hashlib
import numpy as np
from sklearn.model_selection import KFold
from round2_semantic_stack import fit_probes,probe_scores,small_features
from optimize_round2_linear import fit_pairwise
from round5_program_readout import fit_readout,readout,compatibility
from round4_residual import initial_features,fit_residual
from round2_stack import fit_weights
from export_numpy_trees import export_trees


def repeated(indices,multiplicity):
    indices=np.asarray(indices,dtype=np.int64); m=np.asarray(multiplicity)
    assert m.ndim==1 and np.issubdtype(m.dtype,np.integer) and (m>=0).all()
    return np.repeat(indices,m[indices])


def fitting_partition(train,held,easy,multiplicity):
    assert not np.intersect1d(train,held).size
    assert not np.intersect1d(easy,held).size
    fitted=repeated(np.r_[train,easy],multiplicity)
    assert len(fitted)>0 and not np.intersect1d(fitted,held).size
    return fitted


def upstream_features(x,y,a,targets,mask,actual,parsed,ti,vi,easy,multiplicity):
    ti=np.asarray(ti); vi=np.asarray(vi)
    assert not np.intersect1d(ti,vi).size
    assert not np.intersect1d(easy,vi).size
    tf=np.zeros((len(ti),4,387),dtype=np.float32)
    pair=np.zeros((len(ti),4),dtype=np.float32); audits=[]
    for fold,(it,iv) in enumerate(KFold(n_splits=3,shuffle=True,random_state=7509).split(ti)):
        train,held=ti[it],ti[iv]
        fitting=fitting_partition(train,held,easy,multiplicity)
        assert not np.intersect1d(fitting,vi).size
        probe=fit_probes(x[fitting],y[fitting],a[fitting])
        semantic=small_features(x[held],a[held],probe_scores(x[held],probe))
        teacher=fit_readout(x[fitting],targets[fitting],mask[fitting],1000.)
        tf[iv]=np.concatenate([semantic,compatibility(actual[held],parsed[held],readout(x[held],teacher))],axis=2)
        ranking_fit=repeated(train,multiplicity)
        _,pair[iv],_=fit_pairwise(x[ranking_fit],y[ranking_fit],a[ranking_fit],x[held],.01)
        audits.append(dict(fold=fold,unique_training_questions=int(len(np.unique(fitting))),
            effective_training_rows=len(fitting),held_questions=len(held),
            held_question_overlap=0,outer_held_question_overlap=0))
    fitting=fitting_partition(ti,vi,easy,multiplicity)
    probe=fit_probes(x[fitting],y[fitting],a[fitting])
    semantic=small_features(x[vi],a[vi],probe_scores(x[vi],probe))
    teacher=fit_readout(x[fitting],targets[fitting],mask[fitting],1000.)
    vf=np.concatenate([semantic,compatibility(actual[vi],parsed[vi],readout(x[vi],teacher))],axis=2)
    ranking_fit=repeated(ti,multiplicity)
    ranking,vpair,_=fit_pairwise(x[ranking_fit],y[ranking_fit],a[ranking_fit],x[vi],.01)
    return dict(tf=tf,vf=vf,pair=pair,vpair=vpair,probe=probe,teacher=teacher,ranking=ranking,
        audits=audits,unique_training_questions=int(len(np.unique(fitting))),effective_training_rows=len(fitting))


def train_member(x,y,a,targets,mask,actual,parsed,ti,vi,easy,multiplicity,output=None):
    started=time.perf_counter()
    data=upstream_features(x,y,a,targets,mask,actual,parsed,ti,vi,easy,multiplicity)
    positions=np.repeat(np.arange(len(ti)),multiplicity[ti]); assert len(positions)>0
    ft=initial_features(data['pair'],data['tf'][:,:,:294],a[ti])
    fv=initial_features(data['vpair'],data['vf'][:,:,:294],a[vi])
    calibration=fit_weights(ft[positions],y[ti[positions]],a[ti[positions]],.01)
    initial=ft@calibration; vinitial=fv@calibration
    models={}; scores={}
    for name,width in [('compact',294),('structure',387)]:
        model=fit_residual(data['tf'][positions,:,:width],y[ti[positions]],a[ti[positions]],
            initial[positions],dict(leaves=7,l2=10),rounds=120)
        models[name]=model
        scores[name]=model.predict(data['vf'][:,:,:width].reshape(-1,width),num_threads=4,raw_score=True).reshape(-1,4)+vinitial
    blended=.5*scores['compact']+.5*scores['structure']
    if output is not None:
        output.mkdir(parents=True,exist_ok=False)
        np.savez(output/'semantic_probes.npz',**data['probe'])
        np.savez(output/'pair.npz',weight=data['ranking']['weight'])
        np.savez(output/'readout.npz',**data['teacher'])
        np.savez(output/'calibration.npz',weight=calibration)
        for name,model in models.items(): export_trees(model,output/(name+'_trees.npz'))
    record=dict(seconds=time.perf_counter()-started,unique_training_questions=data['unique_training_questions'],
        effective_training_rows=data['effective_training_rows'],residual_unique_questions=int((multiplicity[ti]>0).sum()),
        residual_effective_questions=len(positions),inner_audits=data['audits'],
        multiplicity_sha256=hashlib.sha256(np.asarray(multiplicity,dtype=np.int64).tobytes()).hexdigest(),
        all_learned_stages_resampled=True,held_early_stopping=False,validation_read=False,test_read=False)
    return blended,record,data,initial,vinitial
