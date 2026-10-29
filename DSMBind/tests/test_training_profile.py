import importlib.util
from pathlib import Path
import unittest

import torch


script = Path(__file__).resolve().parents[2] / 'scripts' / 'profile_reference_training.py'
spec = importlib.util.spec_from_file_location('profile_reference_training_test', script)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class TrainingProfileTests(unittest.TestCase):
    def test_profile_starts_at_checkpoint_next_position(self):
        checkpoint = {'training_state': {'indices': [4, 3, 2, 1, 0], 'next_position': 2}}
        self.assertEqual(profile.choose_indices(checkpoint, 2, 5, 7), [2, 1])
        self.assertEqual(profile.choose_indices(checkpoint, 5, 5, 7)[:3], [2, 1, 0])
        with self.assertRaises(ValueError):
            profile.choose_indices({'training_state': {'indices': [20]}}, 1, 5, 7)

    def test_instrumentation_preserves_mixed_derivatives_and_restores_method(self):
        cell = torch.nn.Module()
        cell.weight_c = torch.nn.Parameter(torch.tensor(2.))
        cell.hidden_size = 1
        cell.bidirectional = True
        cell.apply_recurrence = lambda U: U.pow(3) * cell.weight_c
        original = cell.apply_recurrence
        model = torch.nn.Module()
        model.cell = cell
        shapes = {}
        U = torch.tensor(3., requires_grad=True)
        with profile.instrument_recurrences(model, shapes):
            y = cell.apply_recurrence(U)
            force = torch.autograd.grad(y, U, create_graph=True)[0]
            force.square().backward()
        self.assertAlmostEqual(cell.weight_c.grad.item(), 2916.)
        self.assertIs(cell.apply_recurrence, original)
        self.assertEqual(shapes['cell']['device'], 'cpu')
