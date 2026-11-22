"""Run the unchanged official NeXtQSM pipeline in a separate TF environment."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--case", choices=["cosmos", "graz1", "graz2"], required=True)
parser.add_argument("--threads", type=int, default=8)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_NUM_INTRAOP_THREADS"] = str(args.threads)
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
sys.path.insert(0, str(args.reference.resolve()))

import nibabel as nib
import numpy as np
import tensorflow as tf
from nextqsm.predict_all import main
from nextqsm.processing import data_loader
from nextqsm.tf_utils import misc

tf.keras.mixed_precision.set_global_policy("float32")
tf.config.experimental.enable_tensor_float_32_execution(False)
directory = args.output / args.case
source, mask_path = [directory / f"{key}.nii.gz" for key in ("total_field", "mask")]
data, metadata = data_loader.load_testing_volume(dict(source=str(source), mask=str(mask_path)))
loaded = next(iter(data.values()))[0]
for key, path in (("source", source), ("mask", mask_path)):
    expected = nib.load(path).get_fdata(dtype=np.float32)
    assert np.array_equal(loaded[key][0, ..., 0], expected)
    padded, shape, pads = misc.add_padding(expected[None, ..., None], list(expected.shape))
    assert np.array_equal(misc.remove_padding(padded, shape, pads)[0, ..., 0], expected)
checkpoints = args.reference / "nextqsm/checkpoints"
params = json.loads((checkpoints / "params.json").read_text())
assert params["vn_n_steps"] == 6
is_checkout = (args.reference / ".git").is_dir()
commit = (
    subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip()
    if is_checkout else (args.reference / "SOURCE_COMMIT.txt").read_text().strip()
)
provenance = dict(
    source="https://github.com/QSMxT/NeXtQSM",
    commit=commit,
    source_dirty=(bool(subprocess.check_output(["git", "-C", str(args.reference), "diff", "--name-only"], text=True).strip()) if is_checkout else None),
    source_sha256={str(p.relative_to(args.reference)): hashlib.file_digest(p.open("rb"), "sha256").hexdigest() for p in sorted((args.reference / "nextqsm").rglob("*.py"))},
    weights={p.name: hashlib.file_digest(p.open("rb"), "sha256").hexdigest() for p in checkpoints.iterdir() if p.suffix in (".h5", ".npy")},
    lambdas=np.load(checkpoints / "zdir_calc-HR-vn-lambdas.npy").tolist(),
    parameters=params, tensorflow_version=tf.__version__, numpy_version=np.__version__,
    packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    precision="float32", device="CPU", b0_direction=[0,0,1], voxel_size_mm=[1,1,1],
    native_loader_matches_shared_inputs=True, official_padding_roundtrip_exact=True,
)
existing_path = args.output / "nextqsm_provenance.json"
if existing_path.exists():
    existing = json.loads(existing_path.read_text())
    assert existing["commit"] == provenance["commit"]
    assert existing["weights"] == provenance["weights"]
    assert existing["tensorflow_version"] == provenance["tensorflow_version"]
else:
    existing_path.write_text(json.dumps(provenance, indent=2) + "\n")
(directory / "nextqsm_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
print(json.dumps(dict(case=args.case, stage="inference_start", shape=list(loaded["source"].shape))), flush=True)
start = time.perf_counter()
main(str(checkpoints) + "/", "zdir_calc-HR", str(source), str(mask_path), str(directory / "nextqsm_chi.nii.gz"), (0,0,1))
seconds = time.perf_counter()-start
(directory / "nextqsm_chi_BF.nii.gz").rename(directory / "nextqsm_local.nii.gz")
for key in ("chi", "local"):
    image = nib.load(directory / f"nextqsm_{key}.nii.gz")
    assert image.shape == tuple(loaded["source"].shape[1:-1])
    assert image.get_data_dtype() == np.dtype("float32")
    assert np.isfinite(image.get_fdata()).all()
(directory / "nextqsm_runtime.json").write_text(json.dumps(dict(seconds=seconds, device="cpu", dtype="float32", tensorflow_version=tf.__version__), indent=2) + "\n")
print(json.dumps(dict(case=args.case, stage="complete", seconds=seconds)), flush=True)
