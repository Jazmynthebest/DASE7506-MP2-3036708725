# Selecting Reliable LLM Reasoning
**Huang Leyi | 3036708725 | DASE7506 MP2 | 31 October 2026**

**Official full-test accuracy: 84.79657387580299% (396/467).**
The submitted method is a fixed half blend of compact and question-conditioned
arithmetic verifiers. It uses the supplied candidate texts/embeddings and selects
one of the four candidates. Inference requires NumPy only.

## Materials
- `report/MP2_Report_Huang_Leyi.pdf`: final report, at most 10 pages.
- `code/submission.py`: prediction entry point.
- `code/checkpoints/condition_blend/`: all seven inference assets (156,002 bytes).
- `evidence/official/`: official scorer output, predictions, resource measurement and pre-test commitment.
- `EXPERIMENTS.md`, `evidence/report_data.json`: selected development comparisons and ablations.
- `AI_DISCLOSURE.md`: assistance, reuse and data attribution.
- `MODEL_MANIFEST.json`, `FILE_MANIFEST.json`: exact source/checkpoint and release hashes.

## Install and attach the supplied course data
Use Python 3.12 (tested: 3.12.14). Download the complete original course repository
at https://github.com/xudongwu-0/DASE7506-MP2, then locate its `code/data/` folder.
Supplied data is omitted from this small release; its manifest and attribution are retained.
From this release's `code/` directory:

```bash
python -m venv .venv
# PowerShell: .venv/Scripts/Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-inference.txt
python install_course_data.py --source /path/to/original-course-repository/code/data
python check_data.py
python verify_release.py
```

## Reproduce the submitted score without training
```bash
python evaluate_submission.py --split test
```
This launches a fresh prediction process and a fresh independent scoring process,
with four threads. Outputs are written to `code/runs/reproduced-test/`.
Expected `score.json`: `correct=396`, `n=467`, `accuracy=84.79657387580299`.
Recorded local full-test time: 4.350680 seconds on an Intel Core i5-12500H.
Eligibility on the course reference Xeon has not been measured locally.

The wrapper above sets all thread variables automatically. For direct commands,
set them before starting Python.

PowerShell:
```powershell
$env:OMP_NUM_THREADS="4"
$env:OPENBLAS_NUM_THREADS="4"
$env:MKL_NUM_THREADS="4"
$env:NUMEXPR_NUM_THREADS="4"
$env:VECLIB_MAXIMUM_THREADS="4"
$env:BLIS_NUM_THREADS="4"
```

Linux/macOS:
```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 BLIS_NUM_THREADS=4
```

Direct official commands:
```bash
python predict.py --method submission.py --checkpoint checkpoints/condition_blend --split test --output runs/test-predictions.json
python score_predictions.py --split test --predictions runs/test-predictions.json --output runs/test-score.json
```
For a new audit bank, use the original `predict.py` interface with `--split audit`
and `--data-dir /path/to/audit_inputs`. No training or reference files are required
by `submission.py`; evaluation records are never inference inputs.

## Reproduce training
Use a clean working copy. The supplied checkpoint remains the submitted checkpoint.
```bash
python -m pip install -r requirements-method.txt
python train_release.py --output checkpoints/retrained
```
This rebuilds both branches from raw training inputs/references, exports numeric trees,
assembles the fixed 0.5 blend, and compares all learned arrays against the supplied
checkpoint. It does not open validation/test inputs or references. Existing output
directories are rejected. Timing and parameter differences are recorded in
`code/runs/training_reproduction.json`.

Optional nested training-CV reproduction (slower; no development cache needed):
```bash
python reproduce_cv.py --folds 3
python reproduce_cv.py --folds 5
```
Expected compact/arithmetic/blend correct counts: `1774/1776/1782` for three folds,
and `1766/1769/1772` for five folds, out of 2,114 disagreement questions.
Minor floating-point variation across library/OS/BLAS builds is possible; the frozen
checkpoint defines the submitted score. The full-data trainer, rather than these
exploratory outer-fold models, creates the final checkpoint.

## Reproduce report ablations
```bash
python analyze_validation.py
```
This reads only validation inputs/references and the frozen checkpoint. It verifies
the four ablation rows, 74 rescued / 12 introduced plurality errors and five margin
quintiles against the saved report evidence. It performs no training.

## Rebuild the report (optional)
The ready-to-read PDF and editable Markdown are included. For PDF regeneration:
```bash
python -m pip install -r ../tools/requirements-report.txt
python ../tools/build_report.py
```
Regenerating documentation changes the complete-release file hashes. Preserve the
original frozen release for score reproduction, or record new documentation hashes.

## Seeds and scope
Outer CV seeds: 7506 (3 folds), 7591 (5 folds); inner seed: 7509; tree seed: 2026.
Final first-stage cross-fitting seed: 7506. Ridge alpha: 1000; residual trees:
120 rounds, 7 leaves, learning rate 0.025, L2 10. See the report and configuration
for the complete method. `EXPERIMENTS.md` is an editorial selection of key evidence,
not a claim that no other development experiments were performed.

## Attribution
See `AI_DISCLOSURE.md` and `code/data/NOTICE.txt`. Retain the supplied GSM8K MIT license.
The report is a course candidate-selection result, not an original GSM8K generation score.

After obtaining an unmodified release, `python verify_release.py --all` also verifies every published material against `FILE_MANIFEST.json`.
