"""Question-language x arithmetic-relation features; no pretrained language model."""
import re
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from scipy import sparse

STOP = set('a an the is are was were be been being to of and or in on at by for from with as it its this that these those he she they his her their them how what many much does do did has have had will would can could there then than if when who which'.split())


def terms(question):
    words = re.findall(r'[a-z]+',question.lower())
    normalized = []
    for w in words:
        if w.endswith('ies') and len(w)>4:
            w = w[:-3]+'y'
        elif w.endswith('s') and len(w)>3 and not w.endswith('ss'):
            w = w[:-1]
        normalized.append(w)
    singles = [w for w in normalized if w not in STOP]
    pairs = [' '.join(normalized[i:i+2]) for i in range(len(normalized)-1)
             if normalized[i] not in STOP or normalized[i+1] not in STOP]
    return sorted(set(singles+pairs))[:110]


def symbolic_matrix(bank,numeric,n_features=131072):
    def rows():
        for i,row in enumerate(bank):
            phrases = terms(row['question'])
            for j in range(4):
                raw = numeric[i,j,:261]
                # 126 generic operation/scale relations, independent of question IDs.
                exact = raw[1:252:2]
                near = raw[0:252:2]
                active = np.flatnonzero(exact > 0.5)
                almost = np.flatnonzero((near < 0.03) & (exact < 0.5))
                features = [('relation:'+str(k),1.0) for k in active]
                norm = 1/np.sqrt(max(1,len(phrases)))
                for phrase in phrases:
                    features.extend((phrase+'|R'+str(k),norm) for k in active)
                    features.extend((phrase+'|NEAR'+str(k),0.5*norm) for k in almost)
                features.append(('number_of_exact',float(np.log1p(len(active)))))
                yield features
    hasher = FeatureHasher(n_features=n_features,input_type='pair',alternate_sign=True,dtype=np.float32)
    return hasher.transform(rows()).tocsr()
