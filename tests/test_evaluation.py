import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.io import savemat

from dnqsm.evaluation import (
    center_pad,
    evaluate_checkpoint,
    load_cosmos_case,
    load_sim2_case,
)
from dnqsm.model import DNQSM, ModelConfig
from dnqsm.physics import apply_dipole, dipole_tensor
from dnqsm.training import atomic_save


def cosmos_fixture(root: Path):
    directory = root / "cosmos"
    directory.mkdir()
    factor, pad = 0.2125, 2
    generator = torch.Generator().manual_seed(9)
    chi = torch.randn(1, 1, 12, 12, 12, generator=generator) * 0.03
    mask = torch.zeros_like(chi)
    mask[..., pad:-pad, pad:-pad, pad:-pad] = 1
    chi *= mask
    dipole = dipole_tensor(chi.shape[-3:])
    local = apply_dipole(chi, dipole) * mask
    background = torch.ones_like(chi) * mask * 0.02
    noise = torch.randn(chi.shape, generator=generator) * 0.001 * mask
    # The reference stores the full FFT kernel, not its rFFT half.
    from dnqsm.physics import continuous_dipole_kernel

    saved = dict(
        phase=(local + background + noise) / factor,
        chi=chi / factor,
        mask=mask,
        dipole=torch.from_numpy(continuous_dipole_kernel(chi.shape[-3:])).float()[
            None, None
        ],
        background=background / factor,
    )
    torch.save(saved, directory / "simulation.pt")
    (directory / "config.json").write_text(
        json.dumps(dict(factor=factor, pad=pad, snr=100, bg=10))
    )
    return directory / "simulation.pt", chi, local, noise


def sim2_fixture(root: Path):
    directory = root / "sim2"
    (directory / "volumes").mkdir(parents=True)
    shape = (14, 15, 16)
    mask = np.zeros(shape, dtype=np.float32)
    mask[3:11, 4:12, 2:14] = 1
    chi = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) * 1e-5 * mask
    phase = np.ones((16, 17, 18), dtype=np.float32) * 0.03
    savemat(directory / "Sim2.mat", dict(phase=phase))
    savemat(directory / "mask_final.mat", dict(mask_final=mask))
    affine = np.array(
        [[1, 0, 0, -12], [0, 2, 0, 4], [0, 0, 3, 7], [0, 0, 0, 1]], dtype=float
    )
    nib.save(nib.Nifti1Image(chi, affine), directory / "volumes/Sim2ChiGT.nii.gz")
    return directory, chi, mask, affine


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_cosmos_restores_physical_units_and_has_clean_field_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            path, chi, local, noise = cosmos_fixture(Path(temp))
            case = load_cosmos_case(path)
            torch.testing.assert_close(case.chi, chi)
            torch.testing.assert_close(case.local, local)
            self.assertGreater(float((case.phase - case.local - noise).abs().sum()), 0)
            np.testing.assert_allclose(
                case.restore_volume(case.chi),
                chi[0, 0, 2:-2, 2:-2, 2:-2].numpy(),
                rtol=1e-6,
                atol=1e-8,
            )
            self.assertEqual(case.original_shape, (8, 8, 8))
            # Clean-field supervision no longer needs the historical background
            # tensor: it is computed from susceptibility using the forward model.
            saved = torch.load(path, weights_only=True)
            saved.pop("background")
            torch.save(saved, path)
            torch.testing.assert_close(load_cosmos_case(path).local, local)

    def test_sim2_crop_padding_and_affine_restore(self):
        with tempfile.TemporaryDirectory() as temp:
            directory, chi, mask, affine = sim2_fixture(Path(temp))
            case = load_sim2_case(directory, pad=2)
            self.assertEqual(case.phase.shape, (1, 1, 16, 16, 16))
            self.assertEqual(case.pad_offsets, (4, 4, 2))
            self.assertEqual(case.crop_start, (3, 4, 2))
            self.assertEqual(case.crop_shape, (8, 8, 12))
            self.assertIsNone(case.local)
            np.testing.assert_array_equal(case.restore_volume(case.chi), chi)
            np.testing.assert_array_equal(case.restore_volume(case.mask), mask)
            np.testing.assert_array_equal(case.affine, affine)
            np.testing.assert_allclose(case.phase[case.mask.bool()], 0.03)
            padded, _ = center_pad(chi[3:11, 4:12, 2:14], 2)
            np.testing.assert_array_equal(case.chi[0, 0].numpy(), padded)

    def test_full_evaluation_exports_and_reference_availability(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            simulation, *_ = cosmos_fixture(root)
            directory, _, mask, affine = sim2_fixture(root)
            cases = [load_cosmos_case(simulation), load_sim2_case(directory, pad=2)]
            torch.manual_seed(3)
            model = DNQSM(ModelConfig((2, 4)))
            checkpoint = root / "model.pt"
            atomic_save(
                dict(
                    version=1,
                    model_config=asdict(model.config),
                    model=model.state_dict(),
                ),
                checkpoint,
            )
            metrics = evaluate_checkpoint(
                checkpoint, cases, root / "output", "cpu", bf16=False
            )
            self.assertIn("local_nrmse", metrics["cosmos"])
            self.assertNotIn("local_nrmse", metrics["sim2"])
            for name in ("cosmos", "sim2"):
                output = root / "output" / name
                self.assertTrue((output / "comparison.png").is_file())
                self.assertTrue((output / "padded_predictions.pt").is_file())
                self.assertTrue(np.isfinite(metrics[name]["forward_mse"]))
                self.assertGreaterEqual(metrics[name]["chi_nrmse"], 0)
            exported = nib.load(root / "output/sim2/chi.nii.gz")
            self.assertEqual(exported.shape, mask.shape)
            np.testing.assert_array_equal(exported.affine, affine)
            self.assertTrue(np.all(exported.get_fdata()[mask == 0] == 0))
            arrays = np.load(root / "output/sim2/volumes.npz")
            self.assertEqual(arrays["chi"].shape, (8, 8, 12))
            self.assertEqual(
                json.loads((root / "output/sim2/metadata.json").read_text())["factor"],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
