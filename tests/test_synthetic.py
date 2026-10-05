"""CPU synthetic invariants, not real-data or checkpoint reproduction.

Selected definitions are compiled directly from their source AST to avoid
executing dataset/configuration loaders. Bodies are unchanged. JIT decorators
are removed only for NumPy metric tests, which test the Python algorithms.
No pretrained model or patient information is loaded. Optional dependencies
cause explicit skips, never fabricated passing model evaluations.
"""
import ast
import importlib.util
import math
from pathlib import Path
import unittest

try:
    import numpy as np
except ImportError:
    np = None
try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None

ROOT = Path(__file__).resolve().parents[1]


def definitions(relative, names, namespace, strip_decorators=False):
    path = ROOT / relative
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise AssertionError(f"Definitions missing in {relative}: {set(names) - {n.name for n in nodes}}")
    if strip_decorators:
        for node in nodes:
            node.decorator_list = []
    compiled = ast.Module(body=nodes, type_ignores=[])
    env = dict(namespace)
    exec(compile(compiled, str(path), "exec"), env)
    return env


@unittest.skipUnless(np is not None, "NumPy unavailable")
class SamplingTests(unittest.TestCase):
    def test_two_rank_exact_coverage_and_repeatability(self):
        m = definitions("src/student/dataset.py", {"global_batches", "ShardBatches"}, {"np": np})
        for n in (2, 17, 128, 129, 1777):
            a = m["global_batches"](n, 64, 20260919, 1000)
            b = m["global_batches"](n, 64, 20260919, 1000)
            np.testing.assert_array_equal(np.concatenate(a), np.concatenate(b))
            ranks = [list(m["ShardBatches"](a, rank)) for rank in (0, 1)]
            self.assertEqual(len(ranks[0]), len(ranks[1]))
            seen = np.concatenate([x for rank in ranks for x in rank])
            np.testing.assert_array_equal(np.sort(seen), np.arange(n))
            self.assertEqual(len(set(seen)), n)
            self.assertTrue(all(len(batch) > 0 for rank in ranks for batch in rank))


@unittest.skipUnless(torch is not None and np is not None, "PyTorch/NumPy unavailable")
class TorchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(1)

    def adapters(self):
        path = ROOT / "src/student/adapters.py"
        spec = importlib.util.spec_from_file_location("synthetic_adapters", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_parallel_scan_matches_serial_and_gradients(self):
        recurrence = self.adapters().recurrence
        a = torch.sigmoid(torch.randn(2, 9, 3, 4)).requires_grad_()
        b = torch.randn(2, 9, 3, 4, requires_grad=True)
        parallel = recurrence(a, b)
        serial = recurrence(a, b, serial=True)
        torch.testing.assert_close(parallel, serial, atol=2e-6, rtol=2e-6)
        gp = torch.autograd.grad(parallel.sum(), (a, b), retain_graph=True)
        gs = torch.autograd.grad(serial.sum(), (a, b))
        for actual, expected in zip(gp, gs):
            torch.testing.assert_close(actual, expected, atol=4e-6, rtol=4e-6)

    def test_cartesian_polar_adapter_shapes_masks_and_gradients(self):
        adapter = self.adapters().Adapter("cartesian_polar", d=8, n=2)
        x = torch.randn(2, 196, 1024, requires_grad=True)
        disc = torch.tensor([[0., 0., 1.], [.2, -.1, 0.]])
        anatomy = dict(disc=disc, paths=torch.zeros(2, 1, 2, 2),
                       segment_mask=torch.zeros(2, 1, dtype=torch.bool),
                       segment_geometry=torch.zeros(2, 1, 4), adjacency=torch.ones(2, 1, 1))
        y = adapter(x, **anatomy)
        self.assertEqual(tuple(y.shape), (2, 196, 1024))
        self.assertIsNone(adapter.vessel)
        self.assertTrue(torch.isfinite(y).all())
        y.square().mean().backward()
        for fragment in ("core.scans", "polar.radial", "polar.angular"):
            grads = [p.grad for n, p in adapter.named_parameters() if fragment in n and p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)
        # Invalid disc coordinates must not affect the polar-gated output.
        with torch.no_grad():
            changed = disc.clone(); changed[1, :2] = torch.tensor([.9, -.9])
            altered = adapter(x, **dict(anatomy, disc=changed))
            torch.testing.assert_close(altered[1], y[1], atol=0, rtol=0)

    def test_last_four_scope_helper(self):
        env = definitions("src/student/retizero_backbone.py", {"is_trainable_name", "assert_scope"},
                          {"TRAINABLE_BLOCKS": (20, 21, 22, 23)})
        # A tiny synthetic module checks the policy, not pretrained ViT behavior.
        backbone = nn.Module(); backbone.vit = nn.Module()
        backbone.vit.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(24)])
        backbone.vit.norm = nn.LayerNorm(2)
        backbone.requires_grad_(False)
        for block in backbone.vit.blocks[20:]:
            block.requires_grad_(True)
        report = env["assert_scope"](backbone)
        self.assertEqual(report["trainable_blocks_1based"], [21, 22, 23, 24])
        backbone.vit.norm.weight.requires_grad_(True)
        with self.assertRaises(AssertionError):
            env["assert_scope"](backbone)

    def test_multibranch_24_projection_backward(self):
        env = definitions("src/downstream/mlp.py", {"Net"},
                          {"torch": torch, "nn": nn, "np": np, "C": {"branch_hidden": 64, "dropout": .1}})
        dims = {"demo": 3, "z0": 96, "mesh11": 11}
        model = env["Net"](dims, diagnosis=True, prevalence=.2)
        data = {name: torch.randn(4, dim) for name, dim in dims.items()}
        self.assertEqual(model.head.in_features, 72)
        for name in dims:
            self.assertEqual(tuple(model.branches[name](data[name]).shape), (4, 24))
        result = model(data)
        self.assertEqual(tuple(result.shape), (4,))
        nn.functional.binary_cross_entropy_with_logits(result, torch.tensor([0., 1., 0., 1.])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_cox_breslow_ties_and_shift_invariance(self):
        loss_fn = definitions("src/downstream/mlp.py", {"cox_loss"}, {"torch": torch})["cox_loss"]
        r = torch.tensor([.2, -.3, .8, -.1], requires_grad=True)
        t = torch.tensor([2., 2., 4., 5.])
        e = torch.tensor([1., 1., 0., 1.])
        actual = loss_fn(r, t, e)
        reference = -(r[0] - torch.logsumexp(r, 0) + r[1] - torch.logsumexp(r, 0) + r[3] - r[3]) / 3
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(actual, loss_fn(r + 123., t, e), atol=5e-6, rtol=5e-6)
        perm = torch.tensor([3, 1, 0, 2])
        torch.testing.assert_close(actual, loss_fn(r[perm], t[perm], e[perm]))
        actual.backward()
        self.assertTrue(torch.isfinite(r.grad).all())
        self.assertAlmostEqual(float(r.grad.sum()), 0., places=6)
        with self.assertRaises(AssertionError):
            loss_fn(r, t, torch.zeros_like(e))


@unittest.skipUnless(np is not None, "NumPy unavailable")
class MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = definitions("src/downstream/metrics.py",
                            {"div", "binary", "add", "prefixsum", "concordance_sorted", "cargs", "cindex", "ci"},
                            {"np": np}, strip_decorators=True)

    def test_perfect_binary_metrics_and_ties(self):
        y = np.array([0, 1, 0, 1], dtype=bool)
        p = np.array([.1, .9, .2, .8])
        result = self.m["binary"](y, p, np.argsort(-p), np.ones(4), .5)
        np.testing.assert_allclose(result[:8], np.ones(8))
        np.testing.assert_allclose(result[9:], [2, 0, 2, 0])
        p[:] = .5
        tied = self.m["binary"](y, p, np.argsort(-p), np.ones(4), .5)
        self.assertAlmostEqual(tied[0], .5)
        self.assertAlmostEqual(tied[1], .5)

    def test_concordance_perfect_reverse_and_equal(self):
        t = np.array([1., 2., 3.]); e = np.ones(3, dtype=bool)
        for risk, expected in [(np.array([3., 2., 1.]), 1.), (np.array([1., 2., 3.]), 0.), (np.ones(3), .5)]:
            actual = self.m["cindex"](self.m["cargs"](t, e, risk), np.ones(3))
            self.assertAlmostEqual(actual, expected)

    def test_ci_finite_counts_and_no_valid_estimate(self):
        lo, hi, n = self.m["ci"]([0., 1., np.nan])
        self.assertEqual(n, 2)
        self.assertAlmostEqual(lo, .025)
        self.assertAlmostEqual(hi, .975)
        self.assertEqual(self.m["ci"]([np.nan]), (None, None, 0))


if __name__ == "__main__":
    unittest.main()
