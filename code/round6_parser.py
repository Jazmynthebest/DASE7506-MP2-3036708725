"""Conservative fallback for chained numeric equalities in supplied solutions."""
import ast,re
from fractions import Fraction
from common import parse_answer
from features import _clean_math,_calculate
from structure_math import extract_program,question_values,candidate_program


def chained_program(question,text):
    old=candidate_program(question,text)
    if old is not None: return old
    answer=parse_answer(text)
    if answer is None: return None
    cleaned=_clean_math(text)
    cleaned=re.sub(r'\\boxed\{([^{}]*)\}',r'\1',cleaned)
    cleaned=cleaned.replace('$',' ').replace('\\,',' ').replace('\\;',' ')
    cleaned=cleaned.replace('\\!',' ').replace('\\approx','=')
    cleaned=re.sub(r'(\d+(?:\.\d+)?)\s*\\?%',r'((\1)/100)',cleaned)
    # Written decimal percentages are numerically identical to q/100. This
    # substitution is enabled only for numbers explicitly marked as percentages.
    percents=[float(v) for v in re.findall(r'(\d+(?:\.\d+)?)\s*(?:%|percent)',question.lower())]
    def percent_decimal(match):
        value=float(match.group())
        for percentage in percents:
            if 0<abs(value)<1 and abs(value-percentage/100)<1e-8:
                return '('+str(percentage)+'/100)'
        return match.group()
    cleaned=re.sub(r'(?<![\w.])(?:0\.\d+|\.\d+)',percent_decimal,cleaned)
    equations=[]
    for equal in re.finditer('=',cleaned):
        prefix=cleaned[max(0,equal.start()-220):equal.start()]
        leftmatch=re.search(r'[\d.() +*/\-\r\n]+$',prefix)
        if leftmatch is None: continue
        rightmatch=re.match(r'\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:/\d+)?)',cleaned[equal.end():])
        if rightmatch is None: continue
        try: reported=float(Fraction(rightmatch.group(1)))
        except (ValueError,ZeroDivisionError): continue
        expression=leftmatch.group().strip()
        # Numeric list prefixes/prose tails are not arithmetic; try suffixes
        # only at physical newlines. Every retained equality is re-evaluated.
        pieces=[expression]+[expression[m.end():].strip() for m in re.finditer(r'\n',expression)]
        for left in pieces:
            left=re.sub(r'^\d+[.)]\s+','',left)
            if not re.search(r'[+*/\-]',left): continue
            try:
                calculated=_calculate(ast.parse(left,mode='eval'))
                if abs(calculated-reported)>1e-6*max(1,abs(reported)): continue
                equations.append('<<'+left.replace('\n',' ')+'='+str(reported)+'>>')
                break
            except (ValueError,SyntaxError,TypeError,ZeroDivisionError,OverflowError): continue
    return extract_program(question,' '.join(equations)+'\n#### '+answer)
