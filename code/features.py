"""Label-free features. Inference never reads references or uses question IDs."""
import ast
from collections import Counter
from fractions import Fraction
import math
import operator
import re

import numpy as np

from common import parse_answer


NUMBER = re.compile(r"(?<![a-zA-Z])[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
WORDS = re.compile(r"[a-z]+")
CUES = (
    'wait', 'however', 'actually', 'alternatively', 'assume', 'assuming',
    'approximately', 'about', 'round', 'remainder', 'remaining', 'left',
    'total', 'each', 'per', 'half', 'twice', 'double', 'percent', 'difference',
    'more', 'less', 'average', 'ratio', 'profit', 'cost', 'price', 'time',
    'hour', 'minute', 'day', 'week', 'year', 'distance', 'speed', 'area',
    'check', 'verify', 'therefore', 'thus', 'cannot', 'not enough', 'impossible',
    'step', 'answer', 'first', 'second', 'finally', 'let', 'equation',
)
OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
       ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}


def _calculate(node):
    if isinstance(node, ast.Expression):
        return _calculate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _calculate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        a, b = _calculate(node.left), _calculate(node.right)
        if not (abs(a) < 1e12 and abs(b) < 1e12):
            raise ValueError('large expression')
        if isinstance(node.op, ast.Pow) and (abs(b) > 6 or a < 0):
            raise ValueError('unsafe power')
        result = OPS[type(node.op)](a, b)
        if not math.isfinite(result):
            raise ValueError('nonfinite expression')
        return result
    raise ValueError('unsupported arithmetic')


def _clean_math(text):
    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    text = text.replace('\\times', '*').replace('\\cdot', '*').replace('×', '*')
    text = text.replace('\\div', '/').replace('÷', '/').replace('−', '-')
    for _ in range(3):
        text = re.sub(r'\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}', r'((\1)/(\2))', text)
    text = re.sub(r'\\(?:text|mathrm)\{[^{}]*\}', '', text)
    return text.replace('\\left', '').replace('\\right', '').replace('**', '').replace('^', '**')


def arithmetic_features(text):
    """Conservatively check explicit numeric equalities, never execute candidate code."""
    text = _clean_math(text)
    good = bad = 0
    errors = []
    for m in re.finditer(r'([\d.() +*/\-]+)=\s*([-+]?\d+(?:\.\d+)?(?:/\d+)?)', text):
        left, right = m.groups()
        # Ignore numbered-list prefixes and prose fragments; require an operation.
        left = left.strip()
        if not re.search(r'\d\s*[+*/\-]|\)\s*[+*/\-]', left) or len(left) > 160:
            continue
        try:
            value = _calculate(ast.parse(left, mode='eval'))
            target = float(Fraction(right))
            error = abs(value-target) / max(1.0, abs(value), abs(target))
            errors.append(min(error, 1.0))
            if error <= 1e-5:
                good += 1
            else:
                bad += 1
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError):
            continue
    return [math.log1p(good), math.log1p(bad), bad/max(1, good+bad),
            max(errors, default=0), sum(errors)/max(1, len(errors))]


def _numbers(text):
    result = []
    for match in NUMBER.findall(text):
        try:
            value = float(Fraction(match.replace(',', '')))
            if math.isfinite(value) and abs(value) <= 1e15:
                result.append(value)
        except (ValueError, ZeroDivisionError, OverflowError):
            pass
    return result


def _signed_log(value):
    return math.copysign(math.log1p(abs(value)), value)


def answers_for(bank):
    return [[parse_answer(c['text']) for c in row['candidates']] for row in bank]


def majority_choices(bank):
    result = []
    for answers in answers_for(bank):
        counts = Counter(a for a in answers if a is not None)
        result.append(max(range(4), key=lambda j: counts.get(answers[j], 0)))
    return result


def text_features(bank):
    rows = []
    for row in bank:
        question = row['question'].lower()
        qw = set(WORDS.findall(question))
        qnumbers = _numbers(question)
        qabs = [abs(v) for v in qnumbers]
        qmax = max(qabs, default=1)
        texts = [c['text'] for c in row['candidates']]
        answers = [parse_answer(t) for t in texts]
        counts = Counter(a for a in answers if a is not None)
        valid_values = [float(Fraction(a)) for a in answers if a is not None]
        median = float(np.median(valid_values)) if valid_values else 0.0
        cand_rows = []
        for j, text in enumerate(texts):
            low = text.lower()
            words = WORDS.findall(low)
            numbers = _numbers(text)
            answer = answers[j]
            val = float(Fraction(answer)) if answer is not None else 0.0
            val = float(np.clip(val, -1e15, 1e15))
            answer_frequency = counts.get(answer, 0)
            tw = set(words)
            length = max(1, len(text))
            values = [
                float(answer is not None), answer_frequency / 4, len(counts) / 4,
                max(counts.values(), default=0) / 4,
                sum(a is None for a in answers) / 4,
                float(answer_frequency == max(counts.values(), default=0)),
                float(val < 0), float(val == 0), float(val == round(val)),
                _signed_log(val), math.log1p(abs(val)),
                math.log1p(abs(val)/(qmax+1e-6)),
                math.log1p(abs(val-median)),
                float(any(abs(val-v)<1e-8 for v in qnumbers)),
                math.log1p(min([abs(val-v) for v in qnumbers], default=0)),
                math.log1p(length), math.log1p(len(words)), math.log1p(len(numbers)),
                math.log1p(len(set(numbers))), math.log1p(text.count('\n')),
                math.log1p(text.count('=')), math.log1p(text.count('\\boxed')),
                math.log1p(text.count('####')), math.log1p(text.count('$$')),
                math.log1p(text.count('**')), math.log1p(text.count('?')),
                math.log1p(text.count('\\frac') + text.count('\\dfrac')),
                math.log1p(len(question)), math.log1p(len(qnumbers)),
                len(qw & tw)/max(1, len(qw)), len(qw & tw)/max(1, len(tw)),
                len(set(qnumbers)&set(numbers))/max(1, len(set(qnumbers))),
                sum(abs(v-val) < 1e-8 for v in numbers) / max(1, len(numbers)),
                float(text.rstrip().endswith('}')), float('####' in text),
            ]
            values += arithmetic_features(text)
            values += [math.log1p(low.count(cue)) for cue in CUES]
            values += [float(cue in question) for cue in CUES]
            cand_rows.append(values)
        a = np.asarray(cand_rows, dtype=np.float32)
        # Relative features reveal contrasts within a question without IDs.
        rows.append(np.concatenate([a, a-a.mean(axis=0, keepdims=True)], axis=1))
    return np.nan_to_num(np.asarray(rows, dtype=np.float32), posinf=30, neginf=-30)


def make_features(bank, arrays, mode='full'):
    embeddings = np.asarray(arrays['embeddings'], dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    unit = embeddings / np.maximum(norms, 1e-8)
    centered = unit-unit.mean(axis=1, keepdims=True)
    if mode == 'embedding':
        return unit
    if mode == 'relative':
        return np.concatenate([unit, centered], axis=-1)
    text = text_features(bank)
    if mode == 'text':
        return text
    if mode != 'full':
        raise ValueError(f'Unknown feature mode: {mode}')
    similarity = unit @ unit.transpose(0,2,1)
    similarities = []
    for i, answers in enumerate(answers_for(bank)):
        group = []
        for j in range(4):
            same = [k for k in range(4) if k != j and answers[k] is not None and answers[k] == answers[j]]
            other = [k for k in range(4) if k != j and k not in same]
            group.append([float(norms[i,j,0]),
                          float(similarity[i,j].sum()-1)/3,
                          float(max([similarity[i,j,k] for k in same], default=0)),
                          float(max([similarity[i,j,k] for k in other], default=0))])
        similarities.append(group)
    return np.concatenate([unit, centered, text, np.asarray(similarities, dtype=np.float32)], axis=-1)


def choose_scores(bank, scores, pooling=0.0):
    """Optionally pool evidence across candidates with identical numeric answers."""
    scores = np.asarray(scores).reshape(-1,4).copy()
    for i, answers in enumerate(answers_for(bank)):
        for answer in set(answers):
            members = [j for j,a in enumerate(answers) if a == answer]
            if answer is None:
                scores[i,members] = -1e20
            elif pooling:
                v = scores[i,members].copy()
                scores[i,members] = (1-pooling)*v + pooling*v.mean()
    return [int(j) for j in scores.argmax(axis=1)]
