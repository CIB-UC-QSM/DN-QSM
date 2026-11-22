import unittest

import torch

from dnqsm.physics import (
    SPATIAL,
    apply_multiplier,
    cutoff_kernels,
    dipole_tensor,
    truncated_gradient_descent,
)


class TruncatedGradientDescentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.field = torch.randn(
            2, 1, 7, 9, 11, generator=torch.Generator().manual_seed(9)
        )
        self.dipole = dipole_tensor(self.field.shape[-3:])
        _, self.projection = cutoff_kernels(self.dipole, 0.05)
        self.truncated = self.dipole * (1 - self.projection)
        self.step = 1 / float(self.truncated.square().max())

    def test_matches_thirty_spatial_gradient_updates(self):
        reference = torch.zeros_like(self.field)
        for _ in range(30):
            residual = apply_multiplier(reference, self.truncated) - self.field
            reference -= self.step * apply_multiplier(residual, self.truncated)
        actual = truncated_gradient_descent(self.field, self.dipole, 0.05)
        torch.testing.assert_close(actual, reference, rtol=2e-5, atol=3e-6)

    def test_closed_form_response_and_null_band(self):
        inverse, _ = cutoff_kernels(self.dipole, 0.05)
        response = (
            1 - (1 - self.step * self.truncated.double().square()).pow(30)
        ) * inverse.double()
        expected = apply_multiplier(self.field, response)
        actual = truncated_gradient_descent(self.field, self.dipole, 0.05)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
        spectrum = torch.fft.rfftn(actual, dim=SPATIAL)
        relative_null = (spectrum * self.projection).abs().max() / spectrum.abs().max()
        self.assertLess(float(relative_null), 2e-7)

    def test_objective_decreases_and_handles_zero_iterations_and_kernel(self):
        objectives = []
        for count in (0, 1, 2, 5, 30):
            value = truncated_gradient_descent(self.field, self.dipole, 0.05, count)
            objectives.append(
                float(
                    (apply_multiplier(value, self.truncated) - self.field)
                    .square()
                    .mean()
                )
            )
        self.assertTrue(all(a > b for a, b in zip(objectives, objectives[1:])))
        torch.testing.assert_close(
            truncated_gradient_descent(self.field, self.dipole, 0.05, 0),
            torch.zeros_like(self.field),
        )
        torch.testing.assert_close(
            truncated_gradient_descent(self.field, torch.zeros_like(self.dipole), 0.05),
            torch.zeros_like(self.field),
        )

    def test_batched_kernels_fp32_and_gradients(self):
        kernel = self.dipole.repeat(2, 1, 1, 1, 1)
        kernel[1] *= 0.7
        field = self.field.clone().requires_grad_()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = truncated_gradient_descent(field, kernel, 0.05)
        self.assertEqual(output.dtype, torch.float32)
        # Use the same global stable step for the independent sample solves.
        step = 1 / float((kernel * (kernel.abs() > 0.05)).square().max())
        for i in range(2):
            expected = truncated_gradient_descent(
                field[i : i + 1], kernel[i : i + 1], 0.05, step_size=step
            )
            torch.testing.assert_close(output[i : i + 1], expected)
        output.square().mean().backward()
        self.assertTrue(bool(torch.isfinite(field.grad).all()))

    def test_invalid_parameters(self):
        for count in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                truncated_gradient_descent(self.field, self.dipole, 0.05, count)
        for step in (0, -1, float("nan"), float("inf"), 2 * self.step):
            with self.assertRaises(ValueError):
                truncated_gradient_descent(
                    self.field, self.dipole, 0.05, step_size=step
                )
        with self.assertRaises(ValueError):
            truncated_gradient_descent(self.field, self.dipole, 0)


if __name__ == "__main__":
    unittest.main()
