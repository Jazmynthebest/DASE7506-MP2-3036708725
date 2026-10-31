"""Train-reference arithmetic programs and question-conditioned transfer.

Programs are learned only from training reference derivations. No held reference
or answer is consulted at prediction. AST evaluation never executes Python code.
"""
import ast,json,re,math
from fractions import Fraction
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from common import parse_answer
from round2_features import SMALL,TENS

PATTERN=re.compile(r'(?<![\w.])[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:/\d+)?|\b(?:'+
                   '|'.join(sorted(set(SMALL)|set(TENS),key=len,reverse=True))+r')\b',re.I)


def question_values(text):
    values=[]; spans=[]; matches=list(PATTERN.finditer(text.lower())); skip=-1
    for i,m in enumerate(matches):
        if i==skip: continue
        token=m.group().lower()
        if token in SMALL: value=SMALL[token]
        elif token in TENS:
            value=TENS[token]
            if i+1<len(matches) and matches[i+1].group() in SMALL and not text[m.end():matches[i+1].start()].strip(' -'):
                value+=SMALL[matches[i+1].group()]; skip=i+1
        else:
            try: value=float(Fraction(token.replace(',','')))
            except (ValueError,ZeroDivisionError): continue
        if math.isfinite(value) and abs(value)<=1e12:
            values.append(float(value)); spans.append((m.start(),m.end()))
    return values[:16],spans[:16]


def encode_question(text):
    text=text.lower(); numbers,spans=question_values(text)
    parts=[]; position=0
    for i,(start,end) in enumerate(spans):
        parts.append(text[position:start]); parts.append(f' quantity{i} '); position=end
    parts.append(text[position:]); normalized=''.join(parts)
    return normalized+' quantitycount'+str(len(numbers))


def evaluate(program,numbers):
    kind=program[0]
    if kind=='q': return numbers[program[1]]
    if kind=='c': return program[1]
    a=evaluate(program[1],numbers); b=evaluate(program[2],numbers)
    if not(math.isfinite(a) and math.isfinite(b) and abs(a)<=1e12 and abs(b)<=1e12): raise ValueError()
    if kind=='+': value=a+b
    elif kind=='-': value=a-b
    elif kind=='*': value=a*b
    elif kind=='/': value=a/b
    elif kind=='^' and abs(b)<=6 and a>=0: value=a**b
    else: raise ValueError()
    if not math.isfinite(value) or abs(value)>1e12: raise ValueError()
    return value


def extract_program(question,reference):
    numbers,_=question_values(question); previous=[]
    def lookup(value):
        for i,source in enumerate(numbers):
            if abs(value-source)<1e-8*max(1,abs(value)): return ['q',i]
        for source,program in reversed(previous):
            if abs(value-source)<1e-8*max(1,abs(value)): return program
        if value in (0,1,2,3,4,5,6,7,10,12,24,30,60,100,360,365,1000): return ['c',value]
        raise ValueError('unexplained constant')
    def convert(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float): return lookup(float(node.value))
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub): return ['-', ['c',0],convert(node.operand)]
        if isinstance(node,ast.BinOp):
            operator={ast.Add:'+',ast.Sub:'-',ast.Mult:'*',ast.Div:'/',ast.Pow:'^'}.get(type(node.op))
            if operator is None: raise ValueError()
            if isinstance(node.op,ast.Div) and isinstance(node.left,ast.Constant) and isinstance(node.right,ast.Constant):
                value=node.left.value/node.right.value
                if value!=round(value):
                    for i,source in enumerate(numbers):
                        if abs(value-source)<1e-8: return ['q',i]
            a,b=convert(node.left),convert(node.right)
            if operator in ('+','*') and json.dumps(a)>json.dumps(b): a,b=b,a
            return [operator,a,b]
        raise ValueError()
    final=parse_answer(reference)
    if final is None: return None
    target=float(Fraction(final)); candidates=[]
    for expression,result in re.findall(r'<<([^<>]+)=([-+\d.,]+)>>',reference):
        try:
            value=float(result.replace(',','')); expression=expression.replace(',','').replace('^','**')
            program=convert(ast.parse(expression,mode='eval').body)
            if abs(evaluate(program,numbers)-value)>1e-6*max(1,abs(value)): continue
            if abs(target-value)<1e-6*max(1,abs(target)): candidates.append(program)
            previous.append((value,program))
        except (ValueError,SyntaxError,TypeError,ZeroDivisionError,OverflowError,IndexError): continue
    return candidates[-1] if candidates else None


def program_features(queries,answers,train_questions,train_programs):
    """Sparse question similarity votes over symbolic programs, not answer lookups."""
    keep=[i for i,p in enumerate(train_programs) if p is not None]
    vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=24000,sublinear_tf=True,
                             token_pattern=r'(?u)\b[a-z]+\d*\b')
    train=vectorizer.fit_transform([encode_question(train_questions[i]) for i in keep])
    query=vectorizer.transform([encode_question(q) for q in queries])
    programs=[train_programs[i] for i in keep]
    rows=[]
    for i,(q,aa) in enumerate(zip(queries,answers)):
        similarity=(query[i]@train.T).toarray().ravel()
        neighbors=np.argsort(-similarity)[:100]
        numbers,_=question_values(q)
        outputs=np.full(len(neighbors),np.nan)
        for j,k in enumerate(neighbors):
            try: outputs[j]=evaluate(programs[k],numbers)
            except (ValueError,ZeroDivisionError,OverflowError,IndexError): pass
        scores=similarity[neighbors]; finite=np.isfinite(outputs)
        candidate_rows=[]
        for answer in aa:
            try: value=float(Fraction(answer)); match=finite & (np.abs(outputs-value)<=1e-5*max(1,abs(value)))
            except (ValueError,ZeroDivisionError): match=np.zeros(len(outputs),dtype=bool)
            features=[]
            for limit in (3,10,30,100):
                s=scores[:limit]; m=match[:limit]; f=finite[:limit]
                for power in (2,6,12):
                    weight=s**power
                    features.append(float((weight*m).sum()/max((weight*f).sum(),1e-12)))
                features.extend([float(np.max(np.where(m,s,0),initial=0)),float(np.max(np.where(~m & f,s,0),initial=0)),
                                 float(np.mean(m)),float(np.mean(f))])
            features.extend([float(scores[0]),float(scores[:10].mean()),len(numbers)/10])
            candidate_rows.append(features)
        z=np.asarray(candidate_rows,dtype=np.float32)
        rows.append(np.concatenate([z,z-z.mean(0,keepdims=True)],axis=1))
    return np.asarray(rows,dtype=np.float32)
