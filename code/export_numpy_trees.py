"""Export trained numeric trees without changing thresholds, leaves, or order."""
import numpy as np
import lightgbm as lgb


def export_trees(model_file,output):
    booster=model_file if hasattr(model_file,'dump_model') else lgb.Booster(model_file=str(model_file))
    tree_info=booster.dump_model()['tree_info']
    size=max(t['num_leaves'] for t in tree_info)
    n=len(tree_info)
    arrays=dict(feature=np.zeros((n,max(1,size-1)),dtype=np.int32),
                threshold=np.zeros((n,max(1,size-1)),dtype=np.float64),
                left=np.zeros((n,max(1,size-1)),dtype=np.int32),right=np.zeros((n,max(1,size-1)),dtype=np.int32),
                value=np.zeros((n,size),dtype=np.float64),root=np.zeros(n,dtype=np.int32))
    for i,tree in enumerate(tree_info):
        def visit(node):
            if 'leaf_index' in node or 'split_index' not in node:
                j=node.get('leaf_index',0); arrays['value'][i,j]=node['leaf_value']; return -j-1
            j=node['split_index']
            assert node['decision_type']=='<=','Only numeric splits are supported'
            assert node.get('missing_type','None') in ('None','NaN'),'Zero-as-missing requires a different predictor'
            arrays['feature'][i,j]=node['split_feature']; arrays['threshold'][i,j]=node['threshold']
            arrays['left'][i,j]=visit(node['left_child']); arrays['right'][i,j]=visit(node['right_child'])
            return j
        arrays['root'][i]=visit(tree['tree_structure'])
    np.savez(output,**arrays)
    return arrays
