from __future__ import annotations

import random
import unittest
import numpy as np
import torch

from bindenergy.apps.tcr_pmhc.gradient_check import (
    fixed_rng, finite_difference, assess_probe, value_and_gradients, crop_pair,
)


class DetachedBackward(torch.autograd.Function):
    """Deliberately broken mixed derivative, despite a correct first derivative."""
    @staticmethod
    def forward(ctx, coordinates, weight):
        ctx.save_for_backward(coordinates, weight)
        return coordinates * weight

    @staticmethod
    def backward(ctx, upstream):
        coordinates, weight = ctx.saved_tensors
        return (upstream * weight).detach(), (upstream * coordinates).detach()


class GradientCheckTests(unittest.TestCase):
    def closure(self, model, broken=False):
        def evaluate():
            coordinates = torch.tensor(.3, requires_grad=True, dtype=torch.float64)
            energy = (DetachedBackward.apply(coordinates, model.weight[0, 0]) if broken else
                      coordinates * model.weight[0, 0])
            force = torch.autograd.grad(energy, coordinates, create_graph=True)[0]
            signal = force.square() - force
            return {"signal_loss": signal, "loss": signal + .25,
                    "zero_force_loss": force.new_tensor(.25)}
        return evaluate

    def test_finite_difference_detects_missing_mixed_gradient(self):
        model = torch.nn.Linear(1, 1, bias=False).double()
        with torch.no_grad():
            model.weight.fill_(2)
        for broken in (False, True):
            closure = self.closure(model, broken)
            values, grads = value_and_gradients(model, closure)
            derivative = None if grads["weight"] is None else float(grads["weight"].item())
            fd = [finite_difference(model.weight, 0, closure, eps) for eps in (.01, .003)]
            np.testing.assert_allclose(fd, [3, 3], atol=1e-10)
            self.assertEqual(assess_probe(derivative, fd), "fail_missing_gradient" if broken else "pass")
            self.assertEqual(values["loss"], 2.25)
        self.assertEqual(model.weight.item(), 2)

    def test_finite_difference_restores_parameter_after_error(self):
        parameter = torch.nn.Parameter(torch.tensor([2.]))
        def fail():
            raise RuntimeError("test failure")
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            finite_difference(parameter, 0, fail, .1)
        self.assertEqual(parameter.item(), 2)

    def test_flat_and_unstable_probes_are_not_passes(self):
        self.assertEqual(assess_probe(None, [0., 0.]), "inconclusive_low_signal")
        self.assertEqual(assess_probe(1., [1., -2.]), "inconclusive_fd_unstable")
        self.assertEqual(assess_probe(4., [3., 3.]), "fail_gradient_mismatch")

    def test_noise_is_replayed_and_does_not_change_outside_rng(self):
        random.seed(3)
        np.random.seed(3)
        torch.manual_seed(3)
        expected = random.random(), float(np.random.rand()), float(torch.rand(()))
        random.seed(3)
        np.random.seed(3)
        torch.manual_seed(3)
        results = []
        for _ in range(2):
            with fixed_rng(99):
                results.append((random.random(), float(np.random.rand()), float(torch.rand(()))))
        self.assertEqual(results[0], results[1])
        self.assertEqual(expected, (random.random(), float(np.random.rand()), float(torch.rand(()))))

    def test_crop_preserves_corresponding_sequence_and_embedding_indices(self):
        coordinates = torch.randn(6, 14, 3)
        entry = {f"{p}_{k}": value for p in ("binder", "target") for k, value in
                 (("coords", coordinates.clone()), ("seq", "ARNDCQ"), ("idx", torch.arange(6) + 20))}
        cropped = crop_pair(entry, 4)
        self.assertEqual(len(cropped["binder_seq"]), 4)
        for p in ("binder", "target"):
            indices = cropped[f"{p}_idx"] - 20
            self.assertEqual(cropped[f"{p}_seq"], "".join(entry[f"{p}_seq"][i] for i in indices))
            torch.testing.assert_close(cropped[f"{p}_coords"], entry[f"{p}_coords"][indices])
        self.assertEqual(entry["binder_coords"].shape[0], 6)


if __name__ == "__main__":
    unittest.main()
