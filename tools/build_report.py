"""Build a nine-page report and its editable Markdown from frozen evidence."""
import html,json,re
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,KeepTogether
from reportlab.graphics.shapes import Drawing,Rect,Line,String,Polygon

PROJECT=Path(__file__).resolve().parent.parent
RELEASE=PROJECT if (PROJECT/'MODEL_MANIFEST.json').is_file() else PROJECT/'output/final_submission/MP2_3036708725'
NAVY=colors.HexColor('#142E43');TEAL=colors.HexColor('#087F8C');INK=colors.HexColor('#26384A')
PALE=colors.HexColor('#EDF5F7');GRAY=colors.HexColor('#627486');WIDTH=174*mm


def main():
    e=json.loads((RELEASE/'evidence/report_data.json').read_text());manifest=json.loads((RELEASE/'MODEL_MANIFEST.json').read_text())
    training=json.loads((RELEASE/'evidence/training_reproduction.json').read_text())
    report_dir=RELEASE/'report';report_dir.mkdir(exist_ok=True)
    styles={
        'body':ParagraphStyle('body',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=7,textColor=INK),
        'small':ParagraphStyle('small',fontName='Helvetica',fontSize=8.7,leading=12,spaceAfter=6,textColor=GRAY),
        'title':ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=26,leading=30,spaceAfter=13,textColor=NAVY),
        'page':ParagraphStyle('page',fontName='Helvetica-Bold',fontSize=21,leading=25,spaceAfter=12,textColor=NAVY),
        'h':ParagraphStyle('h',fontName='Helvetica-Bold',fontSize=12,leading=16,spaceBefore=10,spaceAfter=6,textColor=TEAL),
        'cell':ParagraphStyle('cell',fontName='Helvetica',fontSize=9,leading=12,textColor=INK),
        'head':ParagraphStyle('head',fontName='Helvetica-Bold',fontSize=9,leading=12,textColor=colors.white),
        'math':ParagraphStyle('math',fontName='Courier',fontSize=8.8,leading=13,textColor=NAVY),
    }
    story=[];md=[]
    def plain(t):return html.unescape(re.sub('<[^>]+>','',t))
    def p(t,small=False):story.append(Paragraph(t,styles['small' if small else 'body']));md.extend([plain(t),''])
    def h(t):story.append(Paragraph(t,styles['h']));md.extend(['## '+t,''])
    def page(t):
        if story:story.append(PageBreak())
        story.append(Paragraph(t,styles['page']));md.extend(['# '+t,''])
    def table(rows,widths):
        cells=[[Paragraph(html.escape(str(v)),styles['head' if i==0 else 'cell']) for v in row] for i,row in enumerate(rows)]
        t=Table(cells,colWidths=[WIDTH*w for w in widths],hAlign='LEFT',repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),TEAL),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F3F6F9')]),
            ('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#C9D7DF'))]))
        story.extend([t,Spacer(1,8)])
        md.append('| '+' | '.join(map(str,rows[0]))+' |');md.append('| '+' | '.join(['---']*len(rows[0]))+' |')
        md.extend('| '+' | '.join(map(str,r))+' |' for r in rows[1:]);md.append('')
    def box(t):
        cell=Table([[Paragraph(t,styles['body'])]],colWidths=[WIDTH]);cell.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,-1),PALE),('LINEBEFORE',(0,0),(0,-1),3,TEAL),
            ('LEFTPADDING',(0,0),(-1,-1),11),('RIGHTPADDING',(0,0),(-1,-1),10),
            ('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),8)]))
        story.extend([cell,Spacer(1,9)]);md.extend(['> '+plain(t),''])
    def formula(t):
        story.append(Paragraph(html.escape(t),styles['math']));story.append(Spacer(1,7));md.extend(['```text',t,'```',''])
    def flow():
        d=Drawing(480,112)
        nodes=[(5,37,82,42,['Four candidates','+ fixed vectors']),
               (101,37,78,42,['Shared linear','evidence']),
               (197,67,90,35,['Compact tree']),
               (197,11,90,35,['Arithmetic tree']),
               (307,37,71,42,['0.5 / 0.5','blend']),
               (397,37,80,42,['Same-answer','pooling'])]
        for x,y,w,hh,lines in nodes:
            d.add(Rect(x,y,w,hh,rx=4,ry=4,fillColor=PALE,strokeColor=colors.HexColor('#AAC8D1'),strokeWidth=.8))
            for k,line in enumerate(lines):d.add(String(x+w/2,y+hh/2+(len(lines)-1)*5-k*11,line,fontName='Helvetica',fontSize=8.5,textAnchor='middle',fillColor=NAVY))
        for x1,y1,x2,y2 in [(87,58,101,58),(179,58,197,84),(179,58,197,28),(287,84,307,58),(287,28,307,58),(378,58,397,58)]:
            d.add(Line(x1,y1,x2,y2,strokeColor=TEAL,strokeWidth=1))
        d.add(String(437,15,'Select index 0-3',fontName='Helvetica',fontSize=8.5,textAnchor='middle',fillColor=GRAY))
        story.extend([d,Spacer(1,3)]);md.extend(['Figure 1. Shared evidence, two residual branches, fixed blending, and answer pooling.',''])
    def margin_chart():
        d=Drawing(480,160);x0=42;y0=27;scale=1.03
        for tick in (0,50,100):
            y=y0+tick*scale;d.add(Line(x0,y,467,y,strokeColor=colors.HexColor('#DCE5EA'),strokeWidth=.5))
            d.add(String(32,y-3,str(tick),textAnchor='end',fontName='Helvetica',fontSize=8,fillColor=GRAY))
        for k,r in enumerate(e['error_analysis']['margin_quintiles']):
            x=58+k*82;v=r['accuracy'];d.add(Rect(x,y0,48,v*scale,fillColor=TEAL,strokeColor=None))
            d.add(String(x+24,y0+v*scale+6,f'{v:.1f}%',textAnchor='middle',fontName='Helvetica-Bold',fontSize=8.5,fillColor=NAVY))
            d.add(String(x+24,12,['Lowest','Q2','Q3','Q4','Highest'][k],textAnchor='middle',fontName='Helvetica',fontSize=8.5,fillColor=GRAY))
        d.add(String(42,149,'Selection accuracy (%) by score-margin quintile',fontName='Helvetica-Bold',fontSize=9,fillColor=NAVY))
        story.extend([d,Spacer(1,3)]);md.extend(['Figure 2. Validation disagreement accuracy by margin quintile; 71 questions per bar.',''])

    p('DASE7506  |  ADVANCED MACHINE LEARNING METHODS  |  MP2',True)
    story.append(Paragraph('Selecting Reliable<br/>LLM Reasoning',styles['title']))
    md.extend(['# Selecting Reliable LLM Reasoning',''])
    p('<b>A compact verifier combining learned candidate quality with arithmetic structure</b>')
    p('Huang Leyi  |  Student ID 3036708725  |  31 October 2026',True)
    box('<b>84.7966% full-test accuracy</b><br/>396 correct selections out of 467 questions; 156,002 bytes of inference assets; 4.35 seconds for fresh prediction plus independent scoring on the local CPU.')
    h('Abstract')
    p('Agreement among candidate solutions is useful evidence, but several explanations can share the same mistaken interpretation. This project tests whether a small learned verifier can improve on answer plurality by combining fixed representations, within-question contrasts, and arithmetic profiles learned from training references. A shared linear stage feeds two compact gradient-boosted residual branches: one scores candidate evidence, while the other also compares an extracted calculation with the arithmetic structure expected from pooled question context. A fixed half blend is followed by mean pooling over identical numeric answers.')
    p('The final frozen method reaches 396/467 on the complete course test bank, compared with 323/467 for plurality and 392/467 for the compact branch. Controlled validation ablations show that both branch combination and answer pooling add two correct selections. These gains are modest, and the method is a final-answer selector rather than a proof checker. The report therefore examines failures, uncertainty and reproducibility alongside accuracy.')
    flow()
    p('Figure 1. Frozen inference pipeline. The two residual branches share the linear prior; answer pooling precedes the candidate choice.',True)
    p('Research question: does question-conditioned arithmetic evidence provide complementary information to learned candidate quality, at a practical CPU and asset cost?',True)

    page('1  Task and evaluation protocol')
    p('Each input question has four supplied candidate traces and a 1,024-dimensional fixed representation for each trace. The method returns one integer index. Correctness is determined solely by the unchanged course numeric parser and reference answer: any candidate with the correct final number receives credit, even if its intermediate reasoning is imperfect.')
    table([['Split','Questions','Disagreement questions','Use'],['Training','4,755','2,114','Parameter fitting / nested CV'],
           ['Validation','798','355','Development selection / analysis'],['Test','467','467','Frozen final scoring']], [.22,.16,.25,.37])
    p('A disagreement question has at least two distinct valid parsed answers. Training and validation also contain groups with one distinct valid answer; there are 443 such validation questions. The selector can resolve these by choosing a valid parse, so full-validation accuracy is easier to obtain than test accuracy. The report always labels the split and denominator.')
    h('Training supervision and cross-fitting')
    p('A training candidate is positive exactly when its canonical numeric answer equals the training reference parse. All four candidates of a question stay together in every split. Three-fold and five-fold outer partitions use seeds 7506 and 7591, stratified by answer diversity and positive-candidate count. First-stage probes and arithmetic readouts for residual-model training are generated by an inner three-fold split with seed 7509. The outer-held question is excluded from every learned stage.')
    p('The residual rankers train on the 2,114 disagreement questions. Easier training groups remain available to fit pointwise embedding probes and the arithmetic readout. At final training, three-fold out-of-fold features train the residual stage; the shared probes/readout are then fitted on all eligible training inputs. No external data, pretrained weights or new generated solutions are added.')
    h('Selection and final commitment')
    p('The compact baseline and arithmetic candidate were compared using training CV and validation. The half blend improves on compact in both recorded training partitions and on validation. Subsequent exploratory variants had to improve the three-fold screen, confirm on the fixed five-fold partition, and then improve validation before promotion. The final selected checkpoint was committed before its own test evaluation; the pre-test commitment and source/asset hashes are included.')
    p('Historical test scores of earlier frozen versions were already known. The final evaluation is therefore not described as a wholly unseen test throughout the project. Its role is to score the previously selected frozen method; the result did not trigger further fitting or tuning. Repeated development on the same training partitions remains exploratory evidence.',True)

    page('2  Shared evidence and compact ranking')
    h('Input representation')
    table([['Feature block','Dimensions','Purpose'],['L2-normalized embedding','1,024','Candidate content'],
           ['Embedding minus question mean','1,024','Within-question contrast'],['Absolute + centered text features','280','Answer validity, agreement, numeric/style evidence'],
           ['Norm and similarity summaries','4','Relations to same/different-answer traces'],['Full representation','2,332','Shared linear ranking input']], [.42,.16,.42])
    p('The text block contains answer frequency, distinct-answer count, sign/magnitude/integrality, relations to numbers in the question, length/formatting statistics, lexical overlap and conservative checks of explicit arithmetic equalities. Formatting is retained as a possible signal, so this representation does not isolate mathematical semantics from writing style. Question IDs are used only for alignment.')
    h('Small embedding probes and linear prior')
    p('Two embedding-only probes provide complementary scores: weighted binary logistic regression (C=0.1) and pairwise logistic ranking (L2=0.01). For each probe, retain its raw score, within-question centered score, mean for the same answer, mean for rival answers, and their difference. Appending these ten values to the 284 non-embedding inputs produces a 294-dimensional compact representation.')
    p('A separate pairwise linear ranker uses the full 2,332 features. Training pairs compare positive candidates against valid negative candidates, with equal total mass per question. A four-term calibration combines normalized full-ranker and probe scores with centered log answer frequency. Content coefficients are nonnegative; the consensus coefficient is bounded in [-2, 2]. Calibration uses training out-of-fold scores and L2=0.01.')
    h('Answer-group objective')
    p('Candidates sharing one numeric answer have the same final-answer correctness. Let G be the set of valid numeric answer groups and s_bar(g) the mean candidate logit within group g. The residual models optimize the group cross-entropy below; duplicates do not create extra classes.')
    formula('L = log(sum_{g in G} exp(s_bar(g))) - s_bar(g_correct)')
    p('The compact residual model has 120 LightGBM trees, seven leaves per tree, learning rate 0.025, minimum leaf size 100, L2=10, feature fraction 0.8 and bagging fraction 0.85. The tree seed is 2026. It adds a small nonlinear correction to the shared linear prior. Tree count is fixed during the final comparisons; held-out early stopping is not used.')

    page('3  Arithmetic structure and fixed fusion')
    h('What should a calculation look like?')
    p('Training reference derivations provide numerical equations. A restricted arithmetic parser maps source quantities to symbolic leaves and chains verified intermediate results. From each extracted program, a 22-dimensional target records operation counts, root operation, source-use/depth statistics and common constants. Reference programs are available for 3,880 of 4,755 training questions.')
    p('A Ridge readout (alpha=1000) predicts this target from the mean of the four normalized candidate embeddings. This mean is a pooled representation of the question context; it is not a separate pretrained question encoder. The readout is fitted only on parsed references inside the applicable training fold. It predicts a structural expectation, not a numeric answer.')
    h('Candidate programs and compatibility')
    p('Candidate programs are extracted from their own numerical equalities and final answers. A checked fallback handles chained equalities and explicit percentages, retaining the original extraction when available. Numerically re-evaluating every accepted equality increases training candidate coverage from 7,338/19,020 (38.58%) to 11,613/19,020 (61.06%). This verifies local arithmetic, not the interpretation of the word problem.')
    p('For each candidate, concatenate the actual 22-vector, expected 22-vector, signed difference and absolute difference. Add five summaries: mean squared discrepancy, mean absolute discrepancy, maximum absolute discrepancy, cosine similarity and parse availability. These 93 features extend the compact inputs from 294 to 387 dimensions. Missing programs are represented explicitly; no candidate is rejected solely because extraction failed.')
    h('Two complementary residual branches')
    p('The arithmetic branch uses the same residual training settings as the compact branch. Both share the embedding probes, full linear prior and calibration. This keeps the comparison focused on the added structural evidence rather than independent large ensembles.')
    formula('raw_j = prior_j + 0.5 * tree_compact_j + 0.5 * tree_arithmetic_j')
    formula('answer_score(g) = mean(raw_j for candidates j in group g)')
    p('Choose a candidate from the highest-scoring valid answer group; exact ties follow candidate order. The fixed 0.5 branch weight was chosen on training CV before its validation evaluation. No question-specific rules, answer lookup tables or cached evaluation predictions enter inference.')
    box('<b>Design boundary.</b> A calculation can be internally consistent while omitting a condition. Arithmetic structure is therefore a soft learned feature. The final selector combines it with content and agreement evidence rather than treating extraction as a proof of correctness.')

    page('4  Main experimental results')
    h('Development comparison: the same questions and partitions')
    table([['Method','Train CV3 / 2,114','Train CV5 / 2,114','Validation / 798'],
           ['Compact branch','1,774','1,766','748'],['Arithmetic branch','1,776','1,769','748'],
           ['Fixed half blend','1,782','1,772','750']], [.34,.22,.22,.22])
    p('The blend improves on compact by eight questions in the three-fold partition and six in the five-fold partition. Each branch alone reaches 748/798 on validation; combining them reaches 750/798. This supports complementary errors, but the two-question validation improvement is small. Both training partitions reuse the same questions and are not independent replications.')
    h('Frozen-model test scores')
    table([['Method','Correct / 467','Accuracy'],['Answer plurality','323','69.1649%'],
           ['Earlier frozen compact model','392','83.9400%'],['Final frozen blend','396','84.7966%']], [.52,.23,.25])
    p('The final method selects 73 more correct answers than plurality and four more than the previously frozen compact checkpoint. The score is the official scorer accuracy on the complete test bank, not validation accuracy, original GSM8K generation accuracy or a verified leaderboard rank. The corresponding prediction file, scorer JSON and measurement JSON are included.')
    h('Validation difficulty')
    table([['Method','All validation','Disagreement only'],['Plurality','688/798 (86.22%)','245/355 (69.01%)'],
           ['Final blend','750/798 (93.98%)','307/355 (86.48%)']], [.34,.33,.33])
    p('On disagreement validation questions, the final selector corrects 74 plurality errors and introduces 12 new errors: a net gain of 62. Plurality ties use the first candidate carrying a most-frequent answer, so this comparison includes both tied and unique-plurality cases. On the complete validation split, easy groups account for much of the higher percentage.')
    p('The report presents selected comparisons that explain the final design. It does not imply that the successful method was the first or only configuration explored. Machine-readable records distinguish model selection from post-freeze validation analysis.',True)

    page('5  Ablations and rejected alternatives')
    h('Fixed-checkpoint validation ablation')
    table([['Setting','All / 798','Disagreement / 355'],
           *[[r['method'],str(r['correct']),str(r['hard_correct'])] for r in e['ablations']]], [.56,.2,.24])
    p('All rows retain the learned weights of the frozen branches. Removing either branch reduces the result from 750 to 748 correct selections. Disabling same-answer mean pooling also reduces it to 748. This inference ablation isolates the decision rule without retraining. It is post-freeze report analysis on validation and did not change the submitted model.')
    p('Pooling makes repeated answers share evidence rather than forcing a choice between equivalent final numbers based on one explanation. The result is two extra correct validation decisions; it does not establish that agreement or every pooled trace is logically reliable.')
    h('Why a slightly better screening score was rejected')
    table([['Method / fixed configuration','CV3','CV5','Decision'],['Frozen blend','1,782','1,772','Retain'],
           ['Contextual pair comparator, 0.25 fusion','1,783','1,767','Reject']], [.51,.13,.13,.23])
    p('A nonlinear comparator reads two candidates, their feature differences, shared context and textual/numeric contrasts. Both comparison orders are included in training, predictions are antisymmetrized, and rival numeric answers receive equal aggregation weight. Its one-question screening gain failed the locked five-fold confirmation. It was therefore not evaluated on validation or test and was not promoted.')
    h('Broader development scope')
    p('Additional exploration covered explicit unit/quantity bindings, reference-language features, small attention models, pipeline resampling, alternative losses, rule-based arithmetic and algebraic sensitivities. Most did not exceed the training-CV screen. A conservative bootstrap blend tied rather than exceeded the baseline; changing the loss could help a minority subgroup while hurting total accuracy. These outcomes favor the smaller frozen design over additional complexity.')
    p('The accompanying experiment summary reports selected families, screened configuration counts and recorded costs. Earlier cached-stage timings overlap and exclude some preprocessing; they are not presented as a complete end-to-end search total. Repeated screening creates selection risk even when no test labels enter fitting.',True)

    page('6  Failure modes and uncertainty')
    h('Score margin provides useful, imperfect ordering')
    margin_chart()
    p('Figure 2 partitions the 355 disagreement validation questions into five equal bins by the margin between the best two distinct-answer scores. Accuracy rises from 49/71 (69.0%) in the lowest-margin bin to 71/71 in the highest. These are descriptive validation bins, not calibrated correctness probabilities or a guarantee of perfect high-confidence behavior on a new bank.',True)
    table([['Correct candidates in group','Questions','Final correct','Plurality correct'],
           *[[str(r['correct_candidates']),str(r['n']),str(r['correct']),str(r['plurality_correct'])] for r in e['by_correct_count']]], [.43,.18,.2,.19])
    h('Correct minority: preserve the requested unit')
    p('A validation problem asks how many minutes faster a 4 mph runner finishes a 24-mile race than a 3 mph runner. Three candidates answer 2 and one answers 120. The selector chooses 120: 24/3 - 24/4 = 2 hours, or 120 minutes. This is behavior beyond plurality, although it does not prove which learned feature caused the choice.')
    h('A remaining missed-condition error')
    p('A training out-of-fold failure asks about four train carriages, each holding 25 seats plus 10 extra passengers, across three trains. The selector chooses 105, omitting the four-carriage factor; the correct answer is 4 x (25+10) x 3 = 420. The selected multiplication is arithmetically valid but solves the wrong quantity. Among 332 training OOF errors, 245 have only one correct candidate, and the compact and arithmetic branches are both wrong on 313.')
    p('Correct answers can also mask faulty reasoning. In a validation plant-growth example, the selected trace reaches the correct 16 while describing an unjustified year-by-year doubling. The scorer properly awards final-answer credit, but such cases prevent interpreting the measured accuracy as stepwise reasoning verification.',True)

    page('7  Efficiency and reproducibility')
    h('Measured resources')
    table([['Item','Measurement / scope'],['Final test result','396/467; 84.79657387580299%'],
           ['Fresh prediction process','3.4745 s'],['Independent scoring process','0.8762 s'],
           ['Complete test evaluation','4.3507 s; four threads'],['Inference assets','156,002 bytes = 0.1488 MiB'],
           ['Local machine','Intel Core i5-12500H; Python 3.12.14; CPU only'],
           ['Clean final training rebuild',f'{training["total_process_seconds"]:.2f} s, including subprocess startup']], [.4,.6])
    p('The local measurement includes startup, imports, supplied data loading, feature extraction, both residual branches and the independent scorer. The course reference Xeon determines eligibility; that reference-machine run has not been measured locally. All learned arrays and configuration files are counted, including both tree ensembles. Supplied bank data and ordinary source code are excluded by the course asset rule.')
    p('Eagerly loading tree arrays once per prediction call avoids repeated NPZ reads while retaining thresholds, precision, traversal and summation order. On three alternating fresh-process validation repetitions, the arithmetic blend median complete time falls from 6.13 to 5.58 seconds. Validation scores, candidate permutations, batches and threshold-boundary checks show exactly identical outputs for this change.')
    h('Reproduction from the released package')
    formula('python evaluate_submission.py --split test')
    formula('python train_release.py --output checkpoints/retrained')
    p('Inference needs only NumPy 2.3.5. Training additionally pins SciPy 1.16.3, scikit-learn 1.7.2, LightGBM 4.6.0 and threadpoolctl 3.6.0. The data-install helper checks the original course manifest. Source, checkpoint and complete-release hashes identify the submission independently of a future Git commit.')
    p('In a clean copy, the raw-data training command rebuilds both branches and reproduces all six learned NPZ parameter files exactly (maximum array difference 0). The checkpoint remains the definition of the submitted score; retraining and optional nested CV are separate verification paths. Neither inference nor raw-data training requires the development cache.')
    p('Release identity: '+manifest['version']+'. The repository README gives complete installation, Windows/Unix thread settings, audit-bank usage and expected outputs.',True)

    page('8  Interpretation, limitations and attribution')
    h('What the evidence supports')
    p('The strongest result is that learned candidate evidence substantially outperforms answer plurality within a small resource budget. Arithmetic profiles add complementary information: the fixed blend improves both recorded training partitions and validation over the compact baseline. The final score rises by four test questions. That is a useful bank-level improvement, but it is not evidence of a large or universally reliable reasoning gain.')
    p('The method has several limits. Program extraction covers only a subset of candidate traces and checks numerical consistency rather than full semantics. The pooled embedding context may already contain misleading candidate interpretations. Style features can correlate with correctness without representing reasoning. The two-branch blend remains highly correlated on errors, and final-answer supervision cannot distinguish an accidental correct number from a sound derivation.')
    p('Repeated development on a fixed public bank and reused CV partitions can overfit model choices. A separate instructor audit bank would provide stronger evidence of generalization. Future work should prioritize better linking of quantities to requested entities and units, and uncertainty calibration evaluated on a genuinely fresh split, rather than assuming that a larger ensemble will improve accuracy.')
    h('AI assistance and reuse')
    p('OpenAI Codex provided substantive assistance with method exploration, implementation, experiment orchestration, debugging, reproducibility checks, and report drafting/editing. Claims are supported by saved outputs and the independent course scorer. The author is responsible for understanding and checking the submitted work. The frozen selector itself runs offline, without API calls or new LLM generation.')
    p('Official course tools and supplied data are retained with attribution. GSM8K questions/references retain their MIT notice. Candidate traces and fixed representations were prepared by the course using Qwen3-0.6B at the revision recorded in the data manifest; no additional model weights are used.')
    h('References')
    p('[1] DASE7506. MP2: Selecting Reliable LLM Reasoning. Course project guide, Fall 2026. Included in guide/GUIDE.md.',True)
    p('[2] Cobbe et al. Training Verifiers to Solve Math Word Problems. 2021. <link href="https://arxiv.org/abs/2110.14168" color="#087F8C">arXiv:2110.14168</link>. Context for learning to select among mathematical solutions; no external examples or pretrained weights from this work are added.',True)
    p('[3] Ke et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. Advances in Neural Information Processing Systems 30, 2017. <link href="https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html" color="#087F8C">Primary publication</link>. Library used for training the small residual tree models.',True)
    p('Final artifacts: report source; pinned dependencies; raw-data training and CV commands; official scorer JSON; prediction/resource records; selected experiments; AI/reuse disclosure; and source/checkpoint/release hashes.',True)

    def decorate(canvas,doc):
        canvas.saveState();w,hh=A4
        canvas.setStrokeColor(colors.HexColor('#D2DDE5'));canvas.setLineWidth(.5);canvas.line(18*mm,hh-17*mm,w-18*mm,hh-17*mm)
        canvas.setFont('Helvetica',7.8);canvas.setFillColor(GRAY)
        canvas.drawString(18*mm,hh-13*mm,'DASE7506 MP2  |  Reliable LLM Reasoning')
        canvas.drawRightString(w-18*mm,hh-13*mm,'Huang Leyi  |  3036708725')
        canvas.drawString(18*mm,12*mm,'Frozen test accuracy: 84.7966%  |  '+manifest['version'])
        canvas.drawRightString(w-18*mm,12*mm,str(doc.page));canvas.restoreState()
    pdf=report_dir/'MP2_Report_Huang_Leyi.pdf'
    doc=SimpleDocTemplate(str(pdf),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=23*mm,bottomMargin=20*mm,
        title='Selecting Reliable LLM Reasoning - Huang Leyi',author='Huang Leyi',subject='DASE7506 MP2 final report')
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
    (report_dir/'REPORT.md').write_text('\n'.join(md),encoding='utf-8')
    print(json.dumps(dict(pdf=str(pdf),markdown=str(report_dir/'REPORT.md')),indent=2))


if __name__=='__main__':main()
