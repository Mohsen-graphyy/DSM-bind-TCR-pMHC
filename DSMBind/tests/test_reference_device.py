"""Device/gradient checks for the reference wrapper, not the real SRU kernel."""
import sys
import types
import unittest
import warnings
from unittest.mock import patch

import torch

from bindenergy.apps.tcr_pmhc.gradient_check import recurrence_backend


class ReferenceDeviceTests(unittest.TestCase):
    def fixture(self, device):
        cell = torch.nn.Module()
        cell.bias = torch.nn.Parameter(torch.tensor(2., device=device))
        cell.weight_c = torch.nn.Parameter(torch.tensor(1., device=device))
        cell.activation_type = 0
        cell.hidden_size = 1
        cell.bidirectional = False
        cell.has_skip_term = True
        cell.apply_recurrence = lambda *args: 'original'
        model = torch.nn.Module()
        model.cell = cell
        package = types.ModuleType('sru')
        package.__path__ = []
        ops = types.ModuleType('sru.ops')
        def reference(U, residual, V, bias, *args):
            warnings.warn_explicit('Running SRU on CPU with grad_enabled=True. Are you sure?',
                                  UserWarning, 'ops.py', 163, module='sru.ops')
            return [U.pow(3) * bias, U]
        ops.elementwise_recurrence_naive = reference
        return model, cell, {'sru': package, 'sru.ops': ops}

    def run_device(self, device):
        model, cell, modules = self.fixture(device)
        original = cell.apply_recurrence
        audit = []
        U = torch.tensor(3., device=device, requires_grad=True)
        with patch.dict(sys.modules, modules), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with recurrence_backend(model, 'reference', required_device=device, device_audit=audit):
                result = cell.apply_recurrence(U, U, U, None, None, None, None)
                force = torch.autograd.grad(result[0], U, create_graph=True)[0]
                force.square().backward()
            self.assertIs(cell.apply_recurrence, original)
            self.assertEqual(bool(caught), device == 'cpu')
        self.assertEqual(len(audit), 1)
        self.assertTrue(audit[0]['grad_enabled'])
        self.assertEqual(result[0].device, U.device)
        self.assertAlmostEqual(cell.bias.grad.item(), 2916.)

    def test_cpu_reference_preserves_warning_and_mixed_gradient(self):
        self.run_device('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_reference_devices_warning_and_mixed_gradient(self):
        self.run_device('cuda')

    def test_cpu_input_is_rejected_when_cuda_was_requested(self):
        model, cell, modules = self.fixture('cpu')
        original = cell.apply_recurrence
        U = torch.tensor(3., requires_grad=True)
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, 'Refusing device fallback'):
                with recurrence_backend(model, 'reference', required_device='cuda'):
                    cell.apply_recurrence(U, U, U, None, None, None, None)
        self.assertIs(cell.apply_recurrence, original)

    def test_mixed_input_devices_are_rejected(self):
        model, cell, modules = self.fixture('cpu')
        U = torch.tensor(3., requires_grad=True)
        with patch.dict(sys.modules, modules), recurrence_backend(model, 'reference', required_device='cpu'):
            with self.assertRaisesRegex(RuntimeError, 'residual is on meta'):
                cell.apply_recurrence(U, U, torch.empty((), device='meta'), None, None, None, None)
