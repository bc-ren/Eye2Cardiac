# Licensing and externally supplied components

No open-source licence is asserted on behalf of the authors. A licence has not
yet been selected through institutional/authorship review. The absence of a
LICENSE is intentional; publication of this source is not an implied licence
grant.

RetiZero source and its original checkpoint are externally supplied inputs. The
release preserves its LoRA parameterization and checkpoint loading; it does not
vendor the RetiZero/RETFound implementations or weights. Obtain them through the
original authors' distribution channels and comply with their terms. The runtime
imports `clip_modules.modeling.LoraRETFound.lora(pretrained=False, R=8)`; swapping a
generic timm ViT is not checkpoint-equivalent. A copy of a modified dependency
must not be published without checking its upstream licence and notices.

VascX vascular/disc preprocessing is an upstream prerequisite. It is not
redistributed or represented as original Eye2Cardiac software. The prepared
anatomical products must use the same coordinate conventions as the CFP inputs.
Foundation-model comparison pipelines (RETFound and EyeCLIP) belong to separate
experiments and are not included in this minimal source package.

PyTorch, torchvision, NumPy, pandas, SciPy, scikit-learn, Numba, Pillow, PyYAML,
PyArrow, VTK, timm and threadpoolctl retain their upstream licences. requirements.txt
declares dependencies, not a transfer of their intellectual-property rights.

UKB and other cohort data have their own access/use agreements. This repository
contains no permission to redistribute participants, labels, dates, retinal
images, prediction records, trained weights or training-derived mean templates.
Fitted preprocessing arrays and mesh topology/template assets are deliberately
kept external pending the relevant data/licensing review.
