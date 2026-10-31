"""Numeric program extraction and structural features; NumPy and standard library only."""
import ast,json,re,math
from fractions import Fraction
import numpy as np
from common import parse_answer
from features import _clean_math
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

def compatibility(actual,parsed,expected):
    n=len(actual); predicted=np.broadcast_to(expected[:,None,:],actual.shape)
    delta=np.where(parsed[:,:,None],actual-predicted,0)
    dot=(actual*predicted).sum(2)
    denominator=np.maximum(np.sqrt((actual*actual).sum(2)*(predicted*predicted).sum(2)),1e-6)
    summary=np.stack([np.mean(delta*delta,axis=2),np.mean(np.abs(delta),axis=2),np.max(np.abs(delta),axis=2),
                      dot/denominator,parsed.astype(np.float32)],axis=2)
    return np.concatenate([actual,predicted,delta,np.abs(delta),summary],axis=2).astype(np.float32)
