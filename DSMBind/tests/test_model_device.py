from __future__ import annotations

import sys
import types
import unittest
from argparse import Namespace

import torch


if "sru" not in sys.modules:
    fake_sru = types.ModuleType("sru")

    class FakeSRUpp(torch.nn.Module):
        def __init__(self, input_size, hidden_size, projection_size, **kwargs):
            super().__init__()
            self.output_size = hidden_size * (2 if kwargs.get("bidirectional") else 1)

        def forward(self, inputs, **kwargs):
            return inputs[..., : self.output_size], None, None

    fake_sru.SRUpp = FakeSRUpp
    sys.modules["sru"] = fake_sru

from bindenergy.models.energy import AllAtomEnergyModel  # noqa: E402
from bindenergy.models.frame import FrameAveraging  # noqa: E402


class DeviceCompatibilityTests(unittest.TestCase):
    def test_frame_averaging_runs_on_cpu(self):
        module = FrameAveraging()
        coordinates = torch.randn(2, 12, 3)
        mask = torch.ones(2, 12)
        framed, operations, centers = module.create_frame(coordinates, mask)
        self.assertEqual(framed.shape, (16, 12, 3))
        self.assertEqual(operations.shape, (2, 8, 3, 3))
        self.assertEqual(centers.shape, (2, 3))
        self.assertEqual(module.ops.device.type, "cpu")
        self.assertTrue(torch.isfinite(framed).all())

    def test_all_atom_loss_and_backward_run_on_cpu(self):
        args = Namespace(
            hidden_size=256,
            bert_size=2560,
            esm_size=2560,
            depth=1,
            dropout=0.1,
            vocab_size=63,
            threshold=14.0,
        )
        model = AllAtomEnergyModel(args)
        coordinates = torch.randn(1, 2, 14, 3)
        features = torch.randn(1, 2, 2560)
        atom_types = torch.ones(1, 2, 14, dtype=torch.long)
        partner = (coordinates, features, atom_types, None)

        loss = model(partner, partner)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_diagnostic_signal_has_same_loss_up_to_noise_constant(self):
        from bindenergy.apps.tcr_pmhc.gradient_check import fixed_rng
        args = Namespace(hidden_size=256, bert_size=2560, esm_size=2560,
                         depth=1, dropout=0.1, vocab_size=63, threshold=14.0)
        model = AllAtomEnergyModel(args).eval()
        partner = (torch.randn(1, 3, 14, 3), torch.randn(1, 3, 2560),
                   torch.ones(1, 3, 14, dtype=torch.long), None)
        with fixed_rng(37):
            ordinary = model(partner, partner)
        with fixed_rng(37):
            full = model(partner, partner, return_diagnostics=True)
        with fixed_rng(37):
            translation = model(partner, partner, return_diagnostics=True, diagnostic_translation_only=True)
        torch.testing.assert_close(ordinary, full['loss'])
        torch.testing.assert_close(full['loss'], full['signal_loss'] + full['zero_force_loss'])
        torch.testing.assert_close(full['translation_loss'], translation['loss'])
        torch.testing.assert_close(translation['loss'], translation['signal_loss'] + translation['zero_force_loss'])


if __name__ == "__main__":
    unittest.main()
