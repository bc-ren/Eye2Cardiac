"""RetiCardiac: Cartesian/polar transfer with last-four-block RetiZero fine-tuning."""
import importlib.util
import torch
from shared import ROOT, S
from retizero_backbone import RetiZeroLast4, assert_scope

spec = importlib.util.spec_from_file_location('immutable_cartesian_student', ROOT / 'reference/student.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
args_of, ANATOMY = base.args_of, base.ANATOMY


class RetiCardiac(base.RetiCardiacBase):
    def __init__(self, c, prior, scalers, white, arm):
        # Construct the unchanged cardiac modules first, preserving RNG ordering.
        super().__init__(c, prior, scalers, white, arm)
        self.backbone = RetiZeroLast4()

    def forward(self, images, cls, eye_mask, **kwargs):
        assert images.shape[1:] == (2, 3, 224, 224)
        b = len(images)
        valid = eye_mask.flatten()
        tokens = self.backbone(images.flatten(0, 1)[valid])
        full = tokens.new_zeros((2 * b, 197, 1024))
        full[valid] = tokens
        full = full.reshape(b, 2, 197, 1024)
        return super().forward(images=full[:, :, 1:], cls=full[:, :, 0], eye_mask=eye_mask, **kwargs)


# Compatibility alias for historical callers; both names refer to one class.
Student = RetiCardiac


def make_models(c, arm='cartesian_polar'):
    assert arm == 'cartesian_polar'
    from fit_artifacts import artifacts
    from model import CardiacDecoder
    from objectives import Geometry
    p, sc, w = artifacts(S['fold'])
    m = RetiCardiac(c, p, sc, w, arm).cuda()
    assert m.patch_encoder.adapter.polar is not None
    assert m.patch_encoder.adapter.vessel is None
    assert_scope(m.backbone)
    return m, CardiacDecoder(c, sc).cuda(), Geometry(c).cuda()
