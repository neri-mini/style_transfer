"""Regression tests use a tiny feature extractor; pretrained weights are not needed."""
import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
NAMES = ("style_transfer_LBFGS_nomask", "style_transfer_ctrl",
         "style_transfer_multi", "style_transfer_mask")


class TransferTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.vgg = nn.Sequential(
            nn.Conv2d(3, 2, 1), nn.ReLU(),
            *[nn.Identity() for _ in range(28)]
        ).eval().requires_grad_(False)
        self.content = torch.full((1, 3, 8, 8), 0.2)
        self.style = torch.full_like(self.content, 0.6)
        self.mask = torch.zeros(1, 1, 8, 8)
        self.mask[:, :, 2:6, 2:6] = 1

    def run_transfer(self, module):
        content, style = module.to_norm(self.content), module.to_norm(self.style)
        if module.__name__.endswith("_multi"):
            grams = module.extract_style_grams(self.vgg, style)
            return module.transfer(self.vgg, content, [grams, grams], (1., 1e4, 0., 1e-4))
        if module.__name__.endswith("_mask"):
            return module.transfer(self.vgg, content, style, self.mask)
        return module.transfer(self.vgg, content, style)

    def test_real_lbfgs_all_modes(self):
        for name in NAMES:
            with self.subTest(mode=name):
                module = importlib.import_module(name)
                with patch.object(module, "NUM_ITERS", 2), patch.object(module, "LBFGS_MAX_ITER", 2):
                    result = self.run_transfer(module)
                image = result[0]
                record = result[3]
                self.assertTrue(torch.isfinite(image).all())
                self.assertTrue(((image >= 0) & (image <= 1)).all())
                best = result[2] if name.endswith("_nomask") else result[1]
                self.assertAlmostEqual(best, min(row[-1] for row in record), places=6)
                if name.endswith("_mask"):
                    outside = (1 - self.mask).expand_as(image).bool()
                    torch.testing.assert_close(image[outside], self.content[outside])

    def test_rejected_trial_does_not_determine_record_or_best_image(self):
        # The last closure evaluates 0.9, but the optimizer accepts 0.6 and then 0.2.
        class TrialOptimizer:
            def __init__(self, parameters, **kwargs):
                self.gen = parameters[0]
                self.step_index = 0
            def zero_grad(self):
                self.gen.grad = None
            def step(self, closure):
                with torch.no_grad():
                    self.gen.fill_(0.9)
                closure()
                with torch.no_grad():
                    self.gen.fill_((0.6, 0.2)[self.step_index])
                self.step_index += 1

        for name in NAMES:
            with self.subTest(mode=name):
                module = importlib.import_module(name)
                def objective(feats, content_target, style_target, x, *args):
                    total = ((module.to_tensor(x) - 0.6) ** 2).mean()
                    zero = total * 0
                    return (zero, zero, zero, zero, total) if name.endswith("_multi") else (zero, zero, zero, total)
                with patch.object(module, "NUM_ITERS", 2), patch.object(module, "loss_calculate", objective), patch.object(torch.optim, "LBFGS", TrialOptimizer):
                    result = self.run_transfer(module)
                record = result[3]
                if name.endswith("_mask"):
                    expected = self.content * (1 - self.mask) + 0.6 * self.mask
                    expected_loss = ((expected - 0.6) ** 2).mean().item()
                else:
                    expected = torch.full_like(self.content, 0.6)
                    expected_loss = 0.
                torch.testing.assert_close(result[0], expected)
                self.assertAlmostEqual(record[1][-1], expected_loss, places=6)
                best = result[2] if name.endswith("_nomask") else result[1]
                self.assertAlmostEqual(best, expected_loss, places=6)

    def test_mask_background_is_fixed_during_optimization(self):
        module = importlib.import_module("style_transfer_mask")
        outside = (1 - self.mask).expand_as(self.content).bool()
        content = self.content
        checks = []
        class ProbeOptimizer:
            def __init__(self, parameters, **kwargs):
                self.gen = parameters[0]
            def zero_grad(self):
                self.gen.grad = None
            def step(self, closure):
                with torch.no_grad():
                    self.gen.fill_(0.9)
                closure()
                torch.testing.assert_close(self.gen.grad[outside], torch.zeros_like(self.gen.grad[outside]))
                checks.append(True)
        original = module.get_features
        def features(vgg, x, track=False):
            if track:
                torch.testing.assert_close(module.to_tensor(x)[outside], content[outside])
            return original(vgg, x, track)
        with patch.object(module, "NUM_ITERS", 1), patch.object(module, "get_features", features), patch.object(torch.optim, "LBFGS", ProbeOptimizer):
            self.run_transfer(module)
        self.assertTrue(checks)

    def test_zero_iterations_returns_initial_image(self):
        for name in NAMES:
            with self.subTest(mode=name):
                module = importlib.import_module(name)
                with patch.object(module, "NUM_ITERS", 0):
                    result = self.run_transfer(module)
                torch.testing.assert_close(result[0], self.content)
                self.assertEqual(len(result[3]), 1)
                best = result[2] if name.endswith("_nomask") else result[1]
                self.assertEqual(best, result[3][0][-1])

    def test_soft_mask_gram_uses_linear_area_weights(self):
        module = importlib.import_module("style_transfer_mask")
        features = torch.tensor([[[[2., 4.]]]])
        mask = torch.tensor([[[[1., 0.25]]]])
        torch.testing.assert_close(module.gram_matrix_masked(features, mask), torch.tensor([[[6.4]]]))


if __name__ == "__main__":
    unittest.main()
