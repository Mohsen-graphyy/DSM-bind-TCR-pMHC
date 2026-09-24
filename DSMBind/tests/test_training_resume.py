from __future__ import annotations

import random
import unittest
from argparse import Namespace

import numpy as np
import torch

from bindenergy.apps.tcr_pmhc.train import (
    capture_rng_state,
    make_checkpoint,
    restore_rng_state,
)


class TrainingResumeTests(unittest.TestCase):
    def test_rng_state_restores_all_training_generators(self):
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        state = capture_rng_state()
        expected = (random.random(), float(np.random.rand()), torch.rand(3))

        random.random()
        np.random.rand()
        torch.rand(3)
        restore_rng_state(state)
        actual = (random.random(), float(np.random.rand()), torch.rand(3))

        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        torch.testing.assert_close(expected[2], actual[2])

    def test_mid_epoch_checkpoint_records_exact_next_position(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
        training_state = {
            "epoch": 2,
            "indices": [3, 1, 2, 0],
            "next_position": 2,
            "running_loss": 4.5,
            "seen_batches": 1,
        }
        payload = make_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            architecture=Namespace(esm_size=2560),
            args=Namespace(batch_size=2),
            best_validation=1.2,
            stale_epochs=0,
            history=[],
            training_state=training_state,
        )

        self.assertEqual(payload["format_version"], 2)
        self.assertEqual(payload["training_state"], training_state)
        self.assertEqual(payload["epoch"], 2)
        self.assertIn("rng_state", payload)


if __name__ == "__main__":
    unittest.main()
