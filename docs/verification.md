# Code and result verification

Audit date: 6 October 2026. Scope: the original 35-epoch trajectory, selected
epoch 4, seed 20260919. The checks below preceded GitHub publication; no
retraining, threshold tuning or cohort editing was performed. Source identities
are recorded in `provenance.json`.

## Canonical model names

The canonical classes are now **CardiacAE** (teacher) and **RetiCardiac**
(student). Factory calls and name metadata use these names; the historical
teacher registry ID and class imports remain direct aliases. No module
attributes, registered buffers, parameter names, numerical method bodies,
initialization order, losses, splits, seed or scientific hyperparameters changed.
Independent static comparison with the pre-rename source confirms that all model method bodies
are identical; historical downstream arm identifiers remain unchanged.

Post-rename local checks: **20 tests, 15 passed, 5 explicitly skipped** because the local
runtime has no PyTorch. The five new tests verify canonical names, legacy alias
identity, registry/factory dispatch and configuration metadata using compiled
declarations and stand-in superclasses. They do not load private checkpoint
weights or substitute for a full-framework inference check. No post-rename numerical
parity or retraining result is claimed here.

## Completed checks before the naming update

| Check | Scope | Outcome |
| --- | --- | --- |
| Teacher source | Seven unchanged files plus the explicitly parameterized preparation script | Original core SHA-256 hashes match |
| Synthetic/runtime tests | 14 tests on the Linux/PyTorch verification host before the final packaging safeguards | 14 passed |
| Final local tests | 15 tests, including escaped configuration paths and overwrite refusal | 10 passed; 5 PyTorch tests skipped because the local runtime has no PyTorch |
| Student/decoder parity | First two same-day and first two asynchronous test records; original versus relocated code and the identical checkpoint | Latent, retinal features, transport, all 50 mesh frames and Mesh11 exactly equal |
| Backward parity | Five representative parameter groups, no optimizer step | Maximum absolute gradient difference 1.08e-9; tolerance 1e-6 |
| Disease-head replay | Nine own-method arms for diagnosis and incident, 18 combinations; 66,916 cached rows per combination | All pass; maximum score difference 1.20e-6, within declared floating-point tolerance |
| Point-metric recomputation | 1,170 checks against saved predictions/operating points | All pass; maximum absolute difference 1.81e-15 |
| CI provenance | 1,458 interval records traced to their original sources | All pass; bootstrap was not repeated |
| Complete result table | 132 rows, 11 methods, five cohorts; 1,782 metric records | 5,346 point/CI cells and 5,346 count cells match the original sources |
| Independent ratio checks | 1,210 metrics recalculated from confusion counts | All pass |
| CFP-age contract | Seven original student split tables, independently joined to original birth/visit metadata | Model and recomputed CFP ages agree exactly; no missing birth/visit match |

Student parity uses BF16 student inference, FP32 decoding and double-precision
Mesh11 geometry, on fixed manifest positions rather than selected favourable
examples. It is a four-example equivalence test, not a full-cohort image rerun.
The independent age check uses the recorded birth year/month with day 15 and
`(CFP date - estimated birth date).days / 365.2425`; it is not exact-day birth
information. Every CFP date matches an original visit date. Split-table counts
overlap and must not be added together as distinct participant counts.
Downstream replay uses the original fitted feature caches; it does not repeat
retinal extraction for all participants. Its score tolerance is absolute 2e-6
plus relative 2e-5. Point metrics use the source completion's float64 arithmetic.
A first audit attempt using native float32 log-loss arithmetic failed a tight
comparison; that failed attempt is retained, and the correct source arithmetic
was verified without changing labels, scores or thresholds.

The final packaging safeguards additionally bind preflight to shared dependency
source hashes, enforce CFP age and the locked random seed, and safely escape
configuration paths. They do not alter model forward equations. Their last
full server rerun was not performed; the completed numerical
parity and 14-test server results above precede these last guard-only edits.
The final source scanner checks Python syntax, obvious private paths/secrets,
binary artifacts and unresolved import candidates. A clean scan is not a
security certification or proof that every dynamic import is executable.

## Results must be identified by the actual head

The current manuscript workbook is a **67-row display subset**, whereas the
archived complete workbook/CSV contains **132 rows**. All retained numerical
values checked against their sources, but `Ours` is not one fixed combination:

- UKB-Prevalent, Poland and the three UKB-Incident headline estimates use
  **Student**.
- UKB-Dev diagnosis uses **D0 + Student**; its 3-/5-year display uses
  **D0 + Student + RetiZero**, and its 10-year display uses **Student**.
- AI-READI's displayed Ours uses **D0 + Student + RetiZero**.

Therefore use `Method`/`source_arm`, cohort and horizon together, not the display
alias alone. No prospective rule explaining the variable display selection can
be established from these tables. The source release preserves all 21 original
head arms instead of treating the display subset as the complete experiment.

The Student-only framework estimates are:

| Cohort | Outcome | N | Cases/events | AUROC, % (95% CI) |
| --- | --- | ---: | ---: | --- |
| UKB-Prevalent | Diagnosis | 41,225 | 1,046 | 77.8 (76.4–79.1) |
| UKB-Incident | 3 years | 36,248 | 295 | 74.7 (72.0–77.2) |
| UKB-Incident | 5 years | 35,874 | 555 | 73.8 (71.7–75.8) |
| UKB-Incident | 10 years | 30,918 | 1,260 | 74.5 (73.2–76.0) |

Horizon estimates are known-status complete-case AUROC, not IPCW estimates.

## Boundaries retained in the release

- No new full training trajectory, clean-machine installation, complete image
  inference or patient-level bootstrap was performed for this packaging task.
- External-cohort and independent RETFound/EyeCLIP results were source-checked,
  not replayed through those separate image-to-prediction pipelines.
- Numerical agreement does not validate clinical labels, true administrative
  follow-up coverage, missing-covariate assumptions or causal interpretation.
- The same-day selection monitor overlaps student training. Teacher pretraining
  also exposed some later student-test participants. These inherited limitations
  are described in `protocol.md`, not silently removed from the history.
- Only source/configuration/documentation/tests are distributed. Restricted
  manifests, patient predictions, data-derived assets and weights stay external.
- Public distribution still needs an author-approved licence and third-party
  rights review. No licence or unrestricted access to the private inputs is
  implied by this package.
