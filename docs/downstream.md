# Disease-head reproduction

The release preserves the original 35-epoch student's downstream numerical
implementation. It does not create a new split, infer missing labels, refit on
external participants, or turn retrospective history into prospectively dated
disease. Cohort preparation and access to restricted inputs remain prerequisites.

**RetiCardiac** is the student model; **CardiacAE** is its cardiac teacher.
Historical method identifiers such as `Student`, `D0+Student` and their saved
feature keys remain unchanged to preserve exact result/checkpoint lookup. Here
`Student` means the RetiCardiac latent branch, not a different model.

## Frozen features and heads

The selected student is frozen. At each disease participant's CFP visit it exports
`z0` (96 columns), the fine-tuned RetiZero retinal feature (1,024 columns), and
`mesh11` (11 columns). No future CMR date, disease date or temporal-transport input
is fed to the disease network. D0 is `age_cfp, sex_cfp, bsa_cfp`, not CMR-age D0.
Mesh11 excludes **only** ED and ES sphericity indices; ED volumes and wall
thickness remain included. The release geometry helper returns the 11 named fields;
the original locked 11-column ordering is preserved before any head scaling.

Every branch is projected `input -> 64 -> 24`, with GELU and dropout 0.1, then
concatenated and passed to one linear output. Five fixed development folds are
used; medians, means, standard deviations and clinical imputers are fitted within
the fold's training records. Diagnosis uses unweighted binary cross-entropy and
selects each fold's epoch by average precision. Incident training uses full-batch
Cox partial likelihood with Breslow ties and selects by Harrell concordance.
The final fit uses all development records for the median selected fold epoch.
AdamW uses LR 0.001, weight decay 0.001, epoch decay 0.95, maximum 100 epochs,
patience 15 and gradient-norm clipping 5. No hyperparameter search is performed.

The code retains all 21 original head arms (16 ordinary multi-branch arms,
4 D0-offset arms and the fixed ClinicalScore comparator), rather than silently
removing unfavourable arms. `corrected_noaug` is the release configuration; the
historical positive-only and 1:5 branches are not enabled. The paper's displayed
subset does not imply that omitted arms were never evaluated.

## Private input contract

The configuration script stages the source into a **private execution directory**.
Within `student/downstream/`, provide the original locked manifests:

| File | Required fields |
| --- | --- |
| `prepared/recorded_timecovered_SERVER_ONLY.parquet` | `eid`, `retina_date`, `age_cfp`, `sex_cfp`, `bsa_cfp`, their observation flags used by `clinical_raw.py`, `original_split`, `paired_view`, `matched_augmentation_role`, `pre_eligible`, `pre_y`, `pre_fold`, image paths and bilateral vascular fields |
| `prepared/historical_country_SERVER_ONLY.parquet` | The same participant order, CFP dates and age; `incident_eligible`, `event`, `time_years`, `surv_fold`, `in_incident_extension`, plus the common split/input fields |
| `clinical_raw/observed_inputs_SERVER_ONLY.npy` | Optional existing `(N, 8)` array ordered as age at CFP, sex (0 female/1 male), total cholesterol and HDL in mmol/L, SBP and DBP in mmHg, diabetes, current smoker; unknown entries stay NaN |
| `features/<seed>/manifest_SERVER_ONLY.parquet` | Exactly the same `eid`, `retina_date`, `age_cfp`, `sex_cfp`, `bsa_cfp` order as the disease manifest |
| `features/<seed>/{z0,retina,mesh11}_SERVER_ONLY.npy` | Existing feature arrays, or outputs of `export_features.py`; never substitute original frozen-RetiZero or RetiDINO caches |

`original_split` is `train`, `validation`, `test` or `evaluation_only`. Five-fold
indices are the original patient-level `0..4`, **not** random regenerated folds.
The two manifests have the same full row order even when task eligibility differs.
Feature and clinical stages require a genuine completed `STATUS.json`, not a
manually invented success marker. `validate_inputs.py` checks overlap, fixed folds,
CFP-age consistency, image coverage and manifest hashes before writing its marker.
It does not prove clinical endpoint construction or administrative coverage;
those are properties of the supplied, independently audited manifests.

The student workspace also needs its original prepared train/test tables,
image manifest, fitted vascular scalers and anatomical products. These contain
restricted data or data-derived artifacts and are deliberately not distributed.
The optional UKB clinical reader selects the latest available assessment **at or
before CFP**, preserving unobserved entries for fold-local imputation. No new
imputation is fitted to any test or external cohort.

## Execution

After configuration and private input staging, run from a shell with `WORK` set to
the private workspace (the release itself is not an experiment output directory):

```bash
python "$WORK/student/downstream/validate_inputs.py"
python "$WORK/student/downstream/clinical_raw.py"
python "$WORK/student/downstream/export_features.py" --seed 20260919
python "$WORK/student/downstream/heads.py" --seed 20260919 --task Diagnosis --variant corrected_noaug
python "$WORK/student/downstream/evaluate.py" --seed 20260919 --task Diagnosis --variant corrected_noaug
python "$WORK/student/downstream/heads.py" --seed 20260919 --task Incident --variant corrected_noaug
python "$WORK/student/downstream/evaluate.py" --seed 20260919 --task Incident --variant corrected_noaug
```

Do not rerun an existing stage: output directories are created exclusively and
failures are marked FAILED or INTERRUPTED. A new experiment needs a new workspace.
Existing trusted completed feature/clinical stages may be staged instead of being
recomputed, but provenance and row alignment must be checked. The original head
implementation requires CUDA. Pickled heads and PyTorch checkpoints must come from
a trusted source; deserialization can execute code.

## Endpoints, operating points and intervals

Diagnosis thresholds maximize Youden's J using development out-of-fold predictions.
Incident thresholds are selected separately at 3, 5 and 10 years using the same
development-only rule. Baseline cumulative hazard is estimated from training
records (Breslow) and frozen for absolute risk inference.

At horizon `h`, eligible evaluation records are events by `h` or participants
followed for at least `h`. Early censored participants are not assigned negative
labels. These are **known-status complete-case** AUROC/AUPRC estimates, not IPCW
cumulative/dynamic discrimination estimates. The source preserves its historical
administrative-censoring assumptions; it does not establish actual ascertainment
coverage. Overall incident evaluation reports Harrell C.

`metrics.py` implements AUROC, average precision (called AUPRC), accuracy,
sensitivity, specificity, PPV, NPV, F1, Brier score and confusion counts.
`evaluate.py` adds balanced accuracy, 2,000 patient-level outcome-stratified
percentile bootstrap intervals and paired differences versus D0 using the same
bootstrap draws. These intervals condition on the already fitted model and are
not between-seed uncertainty. Undefined ratios remain NaN.
`metrics_extended.py` preserves the additional balanced accuracy, MCC and log-loss
formula used in later frozen-prediction completion; it does not tune thresholds.
The original evaluation driver is not silently replaced by a new completion run.

## Comparator and external boundaries

The downstream `RetiZero` branch here is the **last-four-block fine-tuned** encoder.
Independent RETFound and EyeCLIP paper comparators used frozen encoders and
fold-fitted PCA24 with logistic/Cox heads in a separate experiment. They are not
implemented by renaming a branch of this MLP. Their default threshold 0.5 differs
from the Eye2Cardiac development-OOF Youden operating point. This compact release
does not claim to reproduce those independent foundation-model pipelines.

ClinicalScore is the fixed Framingham general CHD 10-year score, computed by the
unchanged `framingham_chd` function. Its use in diagnosis or 3-/5-year tasks is a
rank comparator, not a calibrated probability for those outcomes. ClinicalScore
combinations instead use the same train-fitted branch network as other arms.

For Poland, the locked endpoint is the union of retrospective CHD/MI histories;
exact acquisition-to-diagnosis dates are unavailable. AI-READI uses recorded MI
history as the authorized CHD-proxy endpoint, not an independently trained MI head.
Its sex is missing: the original student prior marginalizes over the prior-training
sex distribution, whereas D0 uses its saved training imputer. These operations
must not be replaced by guessed sex or external-cohort-fitted preprocessing.
Full cohort-specific file parsers and external reporting orchestration are not
bundled here; frozen head inference uses the same `mlp.predict` contract. External
ClinicalScore comparisons also depend on partially missing covariates, handled by
the original training-fitted imputers. Do not describe those as complete clinical
measurement benchmarks.
