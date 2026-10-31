# Experimental evidence

The central hypothesis is that a question-conditioned arithmetic profile adds
information to learned candidate-quality and agreement evidence. All fitting
uses the supplied training split. Question groups remain intact in nested CV.

## Comparisons used for the final architecture

| Method | Train CV3 / 2,114 | Train CV5 / 2,114 | Validation / 798 |
| --- | ---: | ---: | ---: |
| Compact branch | 1,774 | 1,766 | 748 |
| Arithmetic branch | 1,776 | 1,769 | 748 |
| Fixed half blend | 1,782 | 1,772 | 750 |

The three-fold and five-fold partitions reuse the same training questions;
they are confirmations across partitions, not independent new datasets.
The 0.5 fusion weight was selected before the blend's validation evaluation.
The final checkpoint was selected before its own test scoring. Earlier frozen
test results were already known; the report does not claim otherwise.

## Fixed-checkpoint validation ablation

| Inference setting | All validation / 798 | Disagreement / 355 |
| --- | ---: | ---: |
| Compact only | 748 | 305 |
| Arithmetic only | 748 | 305 |
| Blend without same-answer pooling | 748 | 305 |
| Final blend with pooling | 750 | 307 |

These rows are post-freeze report analysis, without refitting. Reproduce with
`python analyze_validation.py` from `code/` after attaching the course data.
They show small complementary gains, not large effects or proof verification.

## Selected rejected families

This is a curated research summary, not an exhaustive chronological diary.
It retains meaningful controls and negative results while omitting repetitive
debugging and equivalent reruns. Earlier exploratory linear, kernel, symbolic
and neural variants are not all itemized here. No claim is made that the final
method was reached on a first attempt, or that every possible improvement is exhausted.

| Family | Screened outputs | Best train CV3 / 2,114 | Decision |
| --- | ---: | ---: | --- |
| Units, target quantities and bindings | 6 | 1779 | No strict screening improvement |
| Reference language, residual resampling and covariance | 12 | 1781 | No strict screening improvement |
| Small neural / attention models with pointwise controls | 24 | 1779 | No strict screening improvement |
| Full-pipeline question bootstrap | 4 | 1782 | No strict screening improvement |
| Question-only arithmetic rules | 4 | 1782 | No strict screening improvement |
| Alternative group objectives | 8 | 1775 | No strict screening improvement |
| Algebraic sensitivity profiles | 8 | 1778 | No strict screening improvement |
| Contextual pair comparators | 4 | 1783 | Failed fixed CV5: 1,767 vs 1,772 |

The 70 outputs across these selected families include controls and multiple
epoch/fusion readouts from the same fitted models; they are not 70 independent
complete model fits. Full metrics and recorded costs for these families are in
`evidence/selected_search_records.json`. The pair comparator's one-question CV3
gain failed its locked CV5 check; it did not reach validation or test.

## Cost record and measurement scope

| Family | Recorded stage seconds | Scope |
| --- | --- | --- |
| Units / bindings | 45.37 feature extraction; 12.89 CV fits | Existing nested base features reused |
| Reference / covariance / residual ensemble | 66.03 preparation; 10.43 and 10.30 residual ensemble fits | Overlapping cached-stage costs; not an experiment total |
| Neural / attention | 121.64 | Training stage; prior features cached |
| Full-pipeline bootstrap | 152.95 | Resampled learned stages; deterministic preprocessing reused |
| Question-only rules | 2.16 feature extraction; 2.09 experiment | Prior pipeline cached |
| Alternative objectives | 12.72 | Residual fitting / scoring; prior pipeline cached |
| Sensitivity profiles | 14.71 extraction; 15.30 experiment | Prior pipeline cached |
| Pair comparator | 9.94 extraction; 6.25 CV3; 8.91 CV5 | Prior pipeline cached |

Do not sum these cached-stage numbers into a purported full-project wall time.
Some stages overlap; earlier feature preparation and human/AI iteration costs
are not fully instrumented. Costs are reported to the scope actually recorded.
All selected experiments used the local Intel Core i5-12500H CPU. The historical
full-data stage records are 18.78 s for compact and 30.41 s for arithmetic,
excluding some preparation/export; see `report_data.json`.

A fresh release rebuild, including four subprocess startups and exports, took
76.05 s and reproduced every saved array exactly (maximum difference 0).
Fresh raw-data CV3/CV5 reproduction took 49.06/69.77 s and recovered all table
counts. These release checks are separate from historical search costs.
The submitted fresh-process full-test measurement is 4.3507 s; an independent
release check recovered 396/467 in 4.8687 s. Local load and OS startup affect timing.

## Seeds, assets and evidence

Outer seeds: 7506 (CV3), 7591 (CV5); nested feature seed: 7509;
final cross-fitting seed: 7506; LightGBM seed: 2026. Complete configuration
and matching source/asset hashes are in `MODEL_MANIFEST.json`.
Assets total 156,002 bytes uncompressed. Reference-Xeon eligibility is not
inferred from the local laptop timing.

The immutable original scoring record is `evidence/official/test-score.json`:
396/467, 84.79657387580299%. New reproduction records are separately named.
No test-row error analysis or fitted test-dependent rules were added while
preparing this report. AI assistance and reuse are described in `AI_DISCLOSURE.md`.
