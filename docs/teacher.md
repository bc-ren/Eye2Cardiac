# CardiacAE teacher

The teacher source derives from the immutable `seed2026_attempt005/source` snapshot used
by the Eye2Cardiac 35-epoch student. It is a non-ODE, factorized periodic
LV/myocardial-surface autoencoder, not the older ODE or four-chamber teacher.
The public class and canonical registry name are `CardiacAE`; the historical
`LVMyoTeacher` import and architecture ID remain aliases for old checkpoints.

## Scope and dependencies

Eight Python files retain the model, data reader, losses, preparation, training,
preflight checks and reporting. Historical orchestration, monitoring and
memory-benchmark scripts are deliberately omitted. The runtime dependencies are
PyTorch, NumPy, pandas, SciPy and PyYAML. The recorded training environment used
PyTorch 2.3.1, CUDA 12.1, two NVIDIA RTX 5000 Ada GPUs, BF16 autocast and TF32.
The reader and training launcher target Linux/CUDA; `read_ed_exact` uses POSIX
file-advice APIs. No graph-learning package or ODE solver is required.

No data, template, topology, checkpoint, cached latent or patient manifest is
distributed. These are required private inputs, not downloads provided by this
source-only release. The scripts do not convert raw DICOM or segment CMR.

## Input and preprocessing contract

The original source manifest has one unique participant per row and columns
`eid`, `split`, `mesh_path`. Its preserved split sizes are 46,969 training,
2,112 validation and 4,204 test participants. The preparation and evaluation
checks deliberately retain these counts: adapting this code to another dataset
requires a separately declared protocol, not silently editing the original
split. Source mesh NPZ files contain finite `mesh_mm` arrays of shape
`(50, 27034, 3)` in millimetres, in the original common coordinate system and
vertex correspondence. The upstream CMR segmentation, mesh fitting and
coordinate-conversion pipeline is not included.

The original topology NPZ must provide `full_labels`, `full_faces` and
`cell_labels`. Labels 1 and 2 identify LV endocardium and myocardial epicardium,
respectively, with 6,141 vertices per surface and matching local face indexing.
Preparation preserves physical size, extracts these surfaces, identifies ED as
the maximum positive signed LV volume and applies the same circular frame shift
to both surfaces. Derived NPY files have C-contiguous float32 shape
`(50, 2, 6141, 3)`, LV first, myocardial outer surface second. It does not perform
subject-specific rescaling or additional registration.

The training-only mean ED mesh is accumulated in manifest order in float64.
`assets/lv_myo_topology.npz` contains this template plus faces, edges, exact
one-ring Laplacian weights, fixed 9-neighbour indices, deterministic
farthest-point downsampling and four-neighbour inverse-distance interpolation.
The hierarchy is 6,141 → 1,536 → 384 → 96 → 24 vertices. No validation or test
mesh contributes to the template. `train_mean_ed_mm.npy` is an audit duplicate;
model construction only requires `lv_myo_topology.npz`.

The encoder divides coordinates by 100 and subtracts the correspondingly scaled
template from ED shape input. Displacement and cyclic velocity are in the same
scaled units. Decoding multiplies learned offsets by 100 and adds the physical
template, returning millimetres. These are fixed unit conversions, not z-score
normalization. Do not substitute a newly fitted template when loading the
published checkpoint: topology and vertex ordering are checkpoint-bound.

## Model and training protocol

The latent comprises a 64-dimensional core and 32-dimensional detail code.
The core has 32 shape and 32 motion dimensions; detail has 16 of each. Each
surface contributes private features, with shared cross-surface features.
Graph decoding uses six Fourier harmonics with the ED-anchored basis
`sin(2πkt)` and `cos(2πkt) − 1`. The output is a periodic 50-frame sequence.
Zero detail exactly recovers core decoding through residual cancellation.

The original planned schedule was A: shape, 20 epochs; B: motion with shape
frozen, 20 epochs; C: joint, 60 epochs. Configuration retains the exact losses,
stage learning rates, warmup, decay, batches, seed 2026 and checkpoint selection
formula. Validation selection balances LV and myocardial geometry:

`J_geom = balanced core RMSE + 0.5 × balanced full RMSE + 0.1 × balanced core P95`.

The actual teacher used downstream is **epoch 84, stage C**, explicitly locked
for downstream use. Training was interrupted at the user's request during
epoch 86. A 100-epoch schedule in the config does **not** mean 100 completed
epochs. Checkpoint SHA-256:

`3c5d623960cf5a6648e736ae2b6d5d4ec7881d782936866451d532eb61332762`.

Original topology SHA-256:

`1efb1c25320b9c75d24ae90456085271818abf4ee22a7d187f95bdca9fa0112b`.

These hashes identify private artifacts; the corresponding files are excluded.
Full-schedule fresh training is not claimed to reproduce the manually locked
epoch-84 checkpoint automatically. The teacher script has no resume CLI.

## Commands with authorized private inputs

Run preparation only when starting from the corresponding original physical
mesh release, not when already holding the derived caches. Outputs must be new:

```bash
python src/teacher/prepare.py \
  --source-manifest /path/to/private/teacher_manifest.csv \
  --source-topology /path/to/private/original_topology.npz \
  --expected-manifest-sha256 YOUR_LOCAL_MANIFEST_SHA256 \
  --root /path/to/new/private_teacher_data --workers 16
```

In the configured private teacher YAML, map `data.root` to that prepared data
directory, `data.manifest` to its `preparation/manifest.csv`, and `data.topology`
to its `assets/lv_myo_topology.npz`. The generated
`preparation/status.json` is required by preflight. A relocated manifest has a
different file hash; verify the remapped participant membership and use its
local hash rather than pretending it is byte-identical to the original file.

After configuring and copying this teacher source into a private working
directory, run the single-GPU then two-GPU preflight. Use the exact same source
directory and YAML for preflight and training:

```bash
python /path/to/workdir/teacher/smoke.py \
  --config /path/to/workdir/teacher/config.yaml \
  --output /path/to/new/private_preflight

torchrun --standalone --nproc_per_node=2 \
  /path/to/workdir/teacher/smoke_ddp.py \
  --config /path/to/workdir/teacher/config.yaml \
  --sanity /path/to/new/private_preflight/sanity.json

torchrun --standalone --nproc_per_node=2 \
  /path/to/workdir/teacher/train.py \
  --config /path/to/workdir/teacher/config.yaml \
  --output-dir /path/to/new/private_teacher_run \
  --seed 2026 --sanity-record /path/to/new/private_preflight/sanity.json
```

Preflight verifies finite losses/gradients, stage training, DDP synchronization,
volume units, periodicity, zero-detail identity, frozen-decoder latent gradients
and checkpoint round-trip. It is a real-data/GPU check, not a synthetic example.
The original DDP preflight also requires at least the configured device-memory
utilization (0.75 in the historical run); this is a hardware-tuning assertion,
not model correctness. On different hardware, explicitly document any batch or
utilization-target change and rerun preflight. Never change the code or config
after preflight: the trainer validates their hashes plus manifest/topology
hashes. Never edit a failed status to `PASS`.

## Interpretation and release boundaries

The myocardial surface is the epicardial boundary, not an independent solid
tissue mesh. The teacher geometry loss is not anatomical validation of clinical
wall thickness. Its original report computes geometric errors and LVEDV,
LVESV and LVEF; downstream Mesh11 measurement code is a separate component.
No new training or private-data evaluation is represented as completed by this
packaging task. Load only trusted PyTorch checkpoints and keep all generated
participant-level outputs outside this public source repository.

In v1, only `prepare.py` was edited during teacher packaging: private source
locations and the fixed manifest-hash literal became explicit mandatory CLI
arguments. In v2, `model.py` additionally exposes the canonical `CardiacAE` class
and registry name, retaining the old names as aliases. All preparation mathematics,
topology construction, model state-dict keys, losses, training reductions and
checkpoint semantics remain unchanged.
