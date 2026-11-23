"""Train the DN-QSM comparison sequentially, then evaluate COSMOS and sim2."""

import argparse
from pathlib import Path
from uuid import uuid4

import torch

from dnqsm.compare import compare
from dnqsm.evaluation import load_cases
from dnqsm.model import DEFAULT_FEATURES
from dnqsm.runtime import require_device
from dnqsm.train_comparison import TrainingConfig, _start_experiment, train_comparison

CUDA_MEMORY_GIB = 90
CHECKPOINT_ROOT = Path("/Workspace/D/DN-QSM")
MOUNTED_CHECKPOINT_ROOT = Path("/WorkSpace/D/DN-QSM")
CALIBRATED_GPU = "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
CALIBRATED_BATCH_SIZES = {
    "model1": 256,
    "local_field": 266,
    "model3_tkd": 295,
    "model3_tikhonov": 295,
    "model3_gd": 295,
    "model4": 295,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, help="Run directory under the checkpoint root"
    )
    parser.add_argument("--cosmos-simulation", type=Path)
    parser.add_argument("--data-dir", type=Path, help="sim2 evaluation input directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--features", type=int, nargs="+", default=DEFAULT_FEATURES)
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument(
        "--epochs", type=int, default=5, help="Full passes per model (default: 5)"
    )
    duration.add_argument(
        "--steps", type=int, help="Run a fixed number of updates per model"
    )
    parser.add_argument("--validation-interval", type=int, default=1_000)
    batching = parser.add_mutually_exclusive_group()
    batching.add_argument(
        "--batch-size", type=int, help="Override batch size for all stages"
    )
    batching.add_argument(
        "--probe-batch-sizes", action="store_true", help="Recalibrate all CUDA stages"
    )
    parser.add_argument("--cuda-memory-gb", type=float, default=CUDA_MEMORY_GIB)
    args = parser.parse_args()

    root = CHECKPOINT_ROOT
    if not root.is_dir() and MOUNTED_CHECKPOINT_ROOT.is_dir():
        root = MOUNTED_CHECKPOINT_ROOT
    root = root.resolve()
    run_id = uuid4().hex
    output = (root / (args.output_dir or f"artifacts/comparison-{run_id}")).resolve()
    if not output.is_relative_to(root):
        parser.error(f"--output-dir must be under {root}")

    device = require_device(args.device)
    if args.probe_batch_sizes and device.type != "cuda":
        parser.error("--probe-batch-sizes requires CUDA")
    batch_sizes = None
    probe_batch_sizes = False
    if device.type == "cuda" and args.batch_size is None:
        if (
            not args.probe_batch_sizes
            and args.cuda_memory_gb == CUDA_MEMORY_GIB
            and tuple(args.features) == DEFAULT_FEATURES
            and torch.cuda.get_device_name(device) == CALIBRATED_GPU
        ):
            batch_sizes = CALIBRATED_BATCH_SIZES
        else:
            probe_batch_sizes = True
    torch.set_num_threads(args.threads)
    cases = load_cases(cosmos_simulation=args.cosmos_simulation, data_dir=args.data_dir)
    output.mkdir(parents=True, exist_ok=False)
    tag = f"comparison-{run_id}"
    training_dir = output / "training"
    train_comparison(
        args.patch_dir,
        training_dir,
        device=device,
        training=TrainingConfig(
            steps=args.steps,
            epochs=None if args.steps is not None else args.epochs,
            validation_interval=args.validation_interval,
            batch_size=1 if args.batch_size is None else args.batch_size,
            workers=args.workers,
        ),
        features=args.features,
        cuda_memory_gb=args.cuda_memory_gb if device.type == "cuda" else None,
        batch_sizes=batch_sizes,
        probe_batch_sizes=probe_batch_sizes,
        comparison_tag=tag,
    )
    experiment = _start_experiment(
        output, "evaluation", tag, archive_dir=training_dir / "comet"
    )
    try:
        compare(
            training_dir / "comparison.json",
            cases,
            output / "evaluation",
            device=device,
            experiment=experiment,
        )
    finally:
        experiment.end()
    print(output / "evaluation" / "summary.json")


if __name__ == "__main__":
    main()
