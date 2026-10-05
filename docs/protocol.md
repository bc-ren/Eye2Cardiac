# Version-locked protocol

This source represents the original `Eye2CardiacMamba_RetiZero_SyncOnly_Last4FT_NoAug_20260927_v1`
experiment, execution `attempt001`, not the later continuation. The 35-epoch
trajectory selected epoch 4, seed 20260919. No new model has been trained while
preparing this release.

The teacher's canonical code name is **CardiacAE** and the student's is
**RetiCardiac**. The v2 naming update does not change the historical run IDs,
trained state dictionaries, data splits or numerical protocol.

## Inputs and fitted artifacts

UKB-Dev same-day training contains 1,777 participants. Each epoch shuffles the
whole pool without replacement, with no padding in the two-rank sampler. The
355-person same-day monitor is a subset of training. The independent asynchronous
validation set contains 1,793 participants. Model selection averages those two
scores equally and requires the original invalid-geometry gate. Same-day test
contains 200 participants; asynchronous test contains 2,341. They do not select
the student checkpoint. The monitor is explicitly apparent, not independent.

The teacher's broader historical fitting exposure remains part of the experiment;
these student tests are not entirely unseen to teacher pretraining. Teacher
selection is the fixed stage-C epoch-84 model. Its training was interrupted at
epoch 86; the configured 100-epoch ceiling is not claimed as completed training.

Images are RGB, bicubic-resized to 224 x 224, with left-eye horizontal reflection
and ImageNet normalization. `spatial_complete` selects the previously processed
RGB image; otherwise the original CFP is read. At least one eye must be present.
The eye mask and ten vascular missingness masks are explicit. The active branch
uses Cartesian/polar selective scans, not the optional vessel-graph scan.

Per-eye VascX fields, in order:

```
temporal_cre_arteries, temporal_cre_veins, temporal_avr,
md_diam_arteries, md_diam_veins,
vd_etdrs_full_arteries, vd_etdrs_full_veins,
mean_sparsity_etdrs_full, md_tort_curv_arteries, md_tort_curv_veins
```

Demographics are `age_cfp, sex_cfp, bsa_cfp`. Sex is coded 0/1, BSA in square
metres. Model age follows the selected CFP visit. Do not substitute diagnosis
age or CMR age at disease inference. The frozen demographic prior was fitted on
40,020 eligible teacher-training participants using CMR-age supervision, then
evaluated at CFP age. Vascular scaling uses the inherited 8,949-person development
pool; residual whitening uses the 1,777 same-day training participants. Thus
same-day-only student optimization does **not** imply all preprocessing was fitted
on only 1,777 participants. Original held-out exclusions and fitted-artifact hashes
are retained; no scaling is re-fitted to evaluation cohorts.

`prepared/` requires seven original patient-level split tables, the image
manifest, and `fold_0/{prior_raw,scalers,residual_whitening}.npz` with its genuine
`AUDIT.json`. See `configs/paths.example.json`. Table columns include `eid`, `view`,
`age_model`, the three demographics, `L_image_path`, `R_image_path`, both eyes'
vascular fields, `latent_raw_000..095`, `cache_path`, `delta_years`, `retina_date`
and `cmr_date`. Targets are physical-mm float32 arrays `(50, 2, 6141, 3)`, LV then
myocardial outer surface. The image manifest contains `image_path, token_id, side,
spatial_complete, preprocessed_rgb`.

Anatomy cache shards `paired_shard0/1` and `clinical_shard0/1` hold `disc, paths,
segment_mask, segment_geometry, adjacency, token_id` NumPy files and genuine
COMPLETED status. The old graph-array interface is retained for compatibility,
although the graph adapter is inactive. Learned retinal features must be freshly
extracted after selecting this fine-tuned RetiZero, not reused from older models.

## RetiCardiac optimization

RetiZero Transformer blocks 21--24 (zero-based 20--23) are fine-tuned, including
their existing LoRA parameters. Earlier blocks, embeddings and final norm remain
frozen. Original LoRA parameterization is not merged or replaced. The demographic
prior and physical cardiac decoder are frozen; the cardiac adapter is trainable.

One AdamW group uses LR 1e-4, weight decay 0.01 and gradient clipping 1. Every
epoch multiplies LR by 0.95. Training lasts 35 epochs with per-GPU batch 64/global
batch 128, two GPUs, BF16 student computation and FP32 physical decoding. Retina
and physical recomputation microbatches are 4; the logical batch and relational
pairs are preserved. No HPO and no positive-case augmentation are enabled.

The physical/latent objectives and coefficients are in `configs/student_model.json`.
Physical supervision is intentionally retained: caching teacher latent alone
does not describe this experiment. Excluding two reported sphericity indices does
not alter the five physical training descriptors. Legacy asynchronous coefficients
remain for exact validation/transport compatibility, but asynchronous samples
contribute **no student-training gradients** in this version. The transport module
received no asynchronous reconstruction supervision. Asynchronous reconstruction
is a known-interval CMR-time estimate, not proven CFP-time ground truth.

## Commands after private configuration

```bash
python "$WORK/student/preflight.py" --gpu-check
torchrun --standalone --nproc_per_node=2 "$WORK/student/train.py" \
  --batch 64 --seed 20260919 --name cartesian_polar_lr1e-04_decay095_seed20260919
```

The source refuses to overwrite a formal run and requires the GPU preflight
fingerprint. The original frozen inputs remain outside the code repository.
Preprocessing regeneration (`fit_artifacts.py`) is a separate, explicit operation;
do not run it against a completed fitted-artifact directory. It rejects an
existing destination. Reusing a trusted existing fit is the normal reproduction
path for this fixed experiment.

Before final test reporting, use `select_checkpoint.py` to record the validation
choice from the completed trajectory; do not pick a test-favourable checkpoint.

```bash
python "$WORK/student/select_checkpoint.py" --run cartesian_polar_lr1e-04_decay095_seed20260919
python "$WORK/student/evaluate_reconstruction.py" \
  --run cartesian_polar_lr1e-04_decay095_seed20260919 --phase sync_only --view same_day
python "$WORK/student/evaluate_reconstruction.py" \
  --run cartesian_polar_lr1e-04_decay095_seed20260919 --phase sync_only --view asynchronous
```

Patient-level files produced by these commands must stay private. Public reports
contain Mesh11 only. Historical 13-field geometry calculations may remain internal
for original QC compatibility; they are not extra published phenotype endpoints.
