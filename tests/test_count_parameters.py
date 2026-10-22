"""Independent counting checks without importing training entry points."""
import importlib.util
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location(
    "count_parameters", Path(__file__).resolve().parents[1] / "src/count_parameters.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ParameterCountTests(unittest.TestCase):
    def test_shared_frozen_parameters_and_buffers(self):
        model = torch.nn.Module()
        model.linear = torch.nn.Linear(3, 2)  # Six weights and two biases.
        model.shared = model.linear
        model.linear.bias.requires_grad_(False)
        model.register_buffer("running", torch.zeros(7))
        self.assertEqual(module.count_parameters(model),
                         {"total": 8, "trainable": 6, "buffers": 7})

    def test_empty_model(self):
        self.assertEqual(module.count_parameters(torch.nn.Identity()),
                         {"total": 0, "trainable": 0, "buffers": 0})


if __name__ == "__main__":
    unittest.main()
