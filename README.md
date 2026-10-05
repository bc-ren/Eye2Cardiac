# Eye2Cardiac

Compact, source-only research release for the **original 35-epoch experiment**.
The selected student is **epoch 4**, seed **20260919**. This is not a later
continuation, a RetiDINO replacement, or the earlier frozen-backbone experiment.

**CardiacAE -> RetiCardiac -> disease heads**

Canonical code names in this v2 package:

| Component | Public class | Source |
| --- | --- | --- |
| Teacher | `CardiacAE` | `src/teacher/model.py` |
| Student | `RetiCardiac` | `src/student/student.py` |

The teacher registry accepts `CardiacAE` and the historical architecture ID;
old `LVMyoTeacher`/`Student` imports remain compatibility aliases. Registered
module/parameter names and checkpoint state dictionaries are unchanged. The
historical downstream arm ID `Student` denotes RetiCardiac latent features and
is retained in saved result keys, not silently relabelled as a new experiment.

- CardiacAE encodes/decodes a 50-frame LV and myocardial-surface sequence into a
  96-dimensional shape/motion representation.
- RetiCardiac uses bilateral CFP, RetiZero last-four-block fine-tuning,
  Cartesian/polar selective scans, vascular measurements and a frozen demographic
  prior. The active model has no vessel-graph adapter.
- Separate diagnosis and Cox incident heads project each selected branch to
  24 dimensions before concatenation. Evaluation includes 3/5/10-year known-status
  complete-case outcomes, **not IPCW time-dependent AUROC**.
- Mesh11 reporting excludes ED and ES sphericity only. Other ED/ES measurements
  and the original training objective remain unchanged.

## Contents

```
src/teacher/       CardiacAE model, preprocessing, losses, training and checks
src/student/       RetiCardiac, exact losses, input transforms and Mesh11
src/student_base/   Small shared prior/decoder/fusion utilities
src/downstream/     Multi-branch diagnosis/Cox training and evaluation
configs/           Version-locked numerical recipes; private-path example
scripts/           Safe workspace configuration and release checks
tests/             Synthetic numerical and static safety checks
docs/              Input contracts, protocol, environment and verification
```

No patient data, real CFP images, participant lists, predictions, fitted arrays,
weights, historical logs, copied foundation-model repositories or Docker images
are included. This is a compact reproducibility source package, **not a standalone
raw-image clinical application or a one-command exact reproduction without
restricted inputs**. The source cohort-building pipelines are not redistributed;
the locked, authorized prepared manifests are required. Do not regenerate random
splits and call the result the original experiment.

## Start here

Use a separate Linux/NVIDIA environment; see [environment](docs/environment.md).

```bash
python scripts/check_release.py
python -m unittest discover -s tests -v
# Copy configs/paths.example.json to a private location and fill real input paths.
python scripts/configure.py --paths /your/private/paths.json --output /your/new/run
```

Configuration only stages code and records hashes; it never launches training.
Then follow [student protocol](docs/protocol.md), [teacher](docs/teacher.md) and
[downstream](docs/downstream.md). Every new formal execution must use a new output
directory. Run real-input/GPU preflight before training. The supplied tests use
synthetic values and cannot establish generalization or clinical validity.

## Version and publication safeguards

RetiCardiac checkpoint SHA-256:
`98020aeaeca21b439cb55473eed1f5bca236f462a79776c2a334e28649ea7589`.

CardiacAE checkpoint SHA-256:
`3c5d623960cf5a6648e736ae2b6d5d4ec7881d782936866451d532eb61332762`.

The same-day selection monitor overlaps training; it is not held-out validation.
Some student test participants were exposed during teacher training. The
asynchronous validation contribution and historical follow-up assumptions are
retained explicitly in [protocol](docs/protocol.md), not silently repaired here.

Use exact `Method` identifiers when comparing saved results. Display names such
as **Ours are not a unique model definition** across all cohort/horizon panels.
Independent RETFound/EyeCLIP comparators are not implemented by renaming this
student's retinal branch. See [verification](docs/verification.md).

This is the source-only GitHub edition of the audited v2 package. No new
training was performed for this publication. An open-source licence has not
been assigned; any licence grant or redistribution of external components still
needs the authors' review. See [third-party notice](THIRD_PARTY.md). Research use
only; not a validated diagnostic device.
