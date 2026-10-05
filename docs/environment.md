# Runtime and reproducibility boundary

Target: Linux x86-64, NVIDIA CUDA, two GPUs for the original DDP training. This
release does not install or modify the host Python/CUDA environment. Create an
isolated environment under the operator's control.

The release verification host reported Python 3.12.7, PyTorch **2.13.0+cu132** and
torchvision **0.28.0+cu132**. Other verified installed versions are pinned in
`requirements.txt`. These are the observed local verification versions, not a
claim that they are the only compatible versions or available in every package
index. Obtain the matching PyTorch/torchvision CUDA build through the vendor's
supported installation process. Do not mix incompatible torchvision binaries.

```bash
# In a user-created isolated environment, after installing matching PyTorch:
python -m pip install -r requirements.txt
python scripts/check_release.py
python -m unittest discover -s tests -v
```

The original CardiacAE teacher run used PyTorch 2.3.1/CUDA 12.1, whereas the current
student release/checkpoint replay host uses the versions above. Teacher source
is preserved; matching original seeds alone does not promise bitwise-identical
retraining across hardware or framework versions. BF16, TF32, CUDA reductions
and DDP may introduce numerical differences. Check real-input forward/backward
and physical-loss chunk equivalence in the target environment before training.

The selective-state-space recurrence is pure PyTorch S6-style code. No official
`mamba-ssm`/VMamba CUDA extension is needed. CardiacAE does not require PyG or
torchdiffeq. Full Mesh11 geometry requires SciPy and VTK. The model uses external
RetiZero source; supplying the original checkpoint alone is insufficient.

The minimal repository intentionally does not freeze the entire historical
Anaconda environment or redistribute foundation-model source, weights and
training-derived templates. No clean-machine GPU installation or full retraining
has been claimed merely from passing the code-level tests. See verification.md
for the actual tested scope and any outstanding limitations.
