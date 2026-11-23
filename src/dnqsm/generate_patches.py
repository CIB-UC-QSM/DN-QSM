"""Generate AHEAD QSM training patches and center-slice QC images."""

import argparse
import json
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from .patch_data import (
    BACKGROUND_OPTIONS,
    PATCH_SIZE,
    SCHEMA_VERSION,
    VARIANTS,
    scale_positive,
)
from .patch_qc import render_center_slices
from .physics import apply_dipole, dipole_tensor

STRIDE = 6
MAX_ZEROS = PATCH_SIZE**3 // 5
PADDING = 32
POSITIVE_SCALE = 4
SPHERE_COUNT = 12
SPHERE_RADII = (4, 12)
SPHERE_CONTRAST = (0.5, 3.0)
SPHERE_CLEARANCE = (2, 20)
DEFAULT_SOURCE = Path("/WorkSpace/D/DN-QSM/data/ahead-original105")
DEFAULT_OUTPUT = Path("/WorkSpace/D/DN-QSM/artifacts/ahead-qsm-patches")


def source_pairs(source_dir):
    images = {p.stem.removeprefix("image_"): p for p in source_dir.glob("image_*.pt")}
    masks = {p.stem.removeprefix("mask_"): p for p in source_dir.glob("mask_*.pt")}
    if images.keys() != masks.keys() or len(images) < 2:
        raise ValueError(
            f"Expected at least two matching source and mask pairs in {source_dir}"
        )
    if not all(source_id.isascii() and source_id.isdecimal() for source_id in images):
        raise ValueError("Source IDs must use decimal digits")
    if len({int(source_id) for source_id in images}) != len(images):
        raise ValueError("Source IDs must identify distinct numeric indices")
    return [(key, images[key], masks[key]) for key in sorted(images, key=int)]


def load_pair(image_path, mask_path):
    chi, mask = (
        torch.load(path, weights_only=True, mmap=True, map_location="cpu").numpy()
        for path in (image_path, mask_path)
    )
    if chi.ndim != 4 or chi.shape[0] != 1 or mask.shape != chi.shape:
        raise ValueError(f"Expected matched [1,D,H,W] arrays in {image_path.name}")
    if not np.isfinite(chi).all() or not np.isin(mask, (0, 1)).all() or not mask.any():
        raise ValueError(f"Invalid susceptibility or mask in {image_path.name}")
    mask = mask[0].astype(bool)
    chi = np.where(mask, chi[0], 0).astype(np.float32)
    return chi, mask


def eligible_starts(chi):
    if min(chi.shape) < PATCH_SIZE:
        raise ValueError(f"Source shape {chi.shape} is smaller than a 64³ patch")
    lo = [np.arange(0, size - PATCH_SIZE + 1, STRIDE) for size in chi.shape]
    hi = [axis + PATCH_SIZE for axis in lo]
    prefix = np.pad(chi != 0, ((1, 0),) * 3)
    for axis in range(3):
        prefix = prefix.cumsum(axis, dtype=np.int32)
    lo0, lo1, lo2 = lo
    hi0, hi1, hi2 = hi
    count = (
        prefix[np.ix_(hi0, hi1, hi2)]
        - prefix[np.ix_(lo0, hi1, hi2)]
        - prefix[np.ix_(hi0, lo1, hi2)]
        - prefix[np.ix_(hi0, hi1, lo2)]
        + prefix[np.ix_(lo0, lo1, hi2)]
        + prefix[np.ix_(lo0, hi1, lo2)]
        + prefix[np.ix_(hi0, lo1, lo2)]
        - prefix[np.ix_(lo0, lo1, lo2)]
    )
    return np.argwhere(count >= PATCH_SIZE**3 - MAX_ZEROS) * STRIDE


def exterior_source(mask, distance, rng):
    baseline = float(rng.uniform(0, 1))
    exterior = np.full(tuple(n + 2 * PADDING for n in mask.shape), baseline, np.float32)
    core = exterior[(slice(PADDING, -PADDING),) * 3]
    core[mask] = 0
    spheres = []
    shape = np.asarray(mask.shape)
    centers_by_radius = {}
    for _ in range(SPHERE_COUNT):
        radius = int(rng.integers(SPHERE_RADII[0], SPHERE_RADII[1] + 1))
        if radius not in centers_by_radius:
            near = np.argwhere(
                (distance > radius + SPHERE_CLEARANCE[0])
                & (distance <= radius + SPHERE_CLEARANCE[1])
            )
            inside = ((near >= radius) & (near < shape - radius)).all(axis=1)
            centers_by_radius[radius] = near[inside]
        candidates = centers_by_radius[radius]
        allowed = np.ones(len(candidates), dtype=bool)
        for sphere in spheres:
            distance_squared = ((candidates - sphere["center"]) ** 2).sum(axis=1)
            allowed &= distance_squared > (radius + sphere["radius"]) ** 2
        candidates = candidates[allowed]
        if not len(candidates):
            raise ValueError(f"Cannot place exterior sphere of radius {radius}")
        center = candidates[rng.integers(len(candidates))].tolist()
        offset = float(rng.uniform(*SPHERE_CONTRAST) * rng.choice((-1, 1)))
        bounds = tuple(slice(c - radius, c + radius + 1) for c in center)
        grid = np.ogrid[tuple(slice(-radius, radius + 1) for _ in center)]
        core[bounds][sum(axis**2 for axis in grid) <= radius**2] += np.float32(offset)
        spheres.append({"center": center, "radius": radius, "offset_ppm": offset})
    return exterior, {"baseline_ppm": baseline, "spheres": spheres}


@lru_cache(maxsize=2)
def simulation_kernel(shape):
    return dipole_tensor(shape)


def project(source, mask):
    kernel = simulation_kernel(source.shape)
    padded_field = apply_dipole(torch.from_numpy(source[None, None]), kernel).numpy()[
        0, 0
    ]
    field = padded_field[(slice(PADDING, -PADDING),) * 3]
    return np.where(mask, field, 0)


def patch_slices(start):
    return tuple(slice(int(x), int(x) + PATCH_SIZE) for x in start)


def store_crops(folder, name, volume, starts, options=None):
    shape = (len(starts),)
    if options is not None:
        shape += (len(options),)
    shape += (PATCH_SIZE,) * 3
    dtype = volume.dtype if options is None else options[0].dtype
    array = np.lib.format.open_memmap(
        folder / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
    )
    for index, start in enumerate(starts):
        bounds = patch_slices(start)
        if options is None:
            array[index] = volume[bounds]
        else:
            for option, field in enumerate(options):
                array[index, option] = field[bounds]
    array.flush()


def generate_source(task):
    pair, split, starts, root, seed, preview, saved_backgrounds = task
    source_id, image_path, mask_path = pair
    chi, mask = load_pair(image_path, mask_path)
    scaled = scale_positive(chi, POSITIVE_SCALE) if split == "train" else None
    local = project(np.pad(chi, PADDING), mask)
    local_scaled = (
        project(np.pad(scaled, PADDING), mask) if scaled is not None else None
    )

    distance = distance_transform_edt(~mask)
    variants = [(VARIANTS[0], chi, local)]
    if scaled is not None:
        variants.extend(
            zip(
                VARIANTS[1:],
                (scaled, -chi, -scaled),
                (local_scaled, -local, -local_scaled),
            )
        )
    fields, parameters = [], []
    if preview:
        best_start = max(
            starts, key=lambda start: np.count_nonzero(chi[patch_slices(start)])
        )
        bounds = patch_slices(best_start)
    for option in range(BACKGROUND_OPTIONS):
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, int(source_id), option])
        )
        exterior, settings = exterior_source(mask, distance, rng)
        if saved_backgrounds is not None and settings != saved_backgrounds[option]:
            raise ValueError(
                f"Background replay disagrees with manifest for {source_id}"
            )
        background = project(exterior, mask)
        exterior = exterior[(slice(PADDING, -PADDING),) * 3]
        fields.append(background)
        parameters.append(settings)
        for name, susceptibility, local_field in variants:
            title = f"Source {source_id} | {split} | {name} | background {option}"
            title += f" | baseline {settings['baseline_ppm']:.3f} ppm"
            volumes = (
                susceptibility,
                mask,
                exterior,
                local_field,
                background,
                local_field + background,
            )
            filename = f"{source_id}_{name}_bg{option}.png"
            render_center_slices(root / "qc" / "volumes" / filename, title, volumes)
            if preview:
                crop_title = f"{title} | patch start {tuple(best_start.tolist())}"
                render_center_slices(
                    root / "qc" / "patches" / filename,
                    crop_title,
                    tuple(v[bounds] for v in volumes),
                )
    if saved_backgrounds is None:
        folder = root / "patches" / source_id
        folder.mkdir(parents=True)
        for name, volume in (
            ("chi", chi),
            ("mask", mask),
            ("local", local),
            ("local_scaled", local_scaled),
        ):
            if volume is not None:
                store_crops(folder, name, volume, starts)
        store_crops(folder, "background", None, starts, fields)
    return {
        "source_id": source_id,
        "image": image_path.name,
        "mask": mask_path.name,
        "split": split,
        "shape": list(chi.shape),
        "starts": starts.tolist(),
        "background_options": parameters,
    }


def process_sources(tasks, workers):
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        torch.set_num_threads(2)
        yield from map(generate_source, tasks)
        return
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=torch.set_num_threads,
        initargs=(2,),
    ) as pool:
        yield from pool.map(generate_source, tasks)


def prepare_qc(root):
    for category in ("volumes", "patches"):
        (root / "qc" / category).mkdir(parents=True)


def refresh_qc(output_dir, *, workers=4):
    """Replay the saved simulations for QC without writing patch arrays."""
    output_dir = Path(output_dir).resolve()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported QSM patch dataset version")
    source_dir = Path(manifest["source_dir"])
    with TemporaryDirectory(prefix=".qc-", dir=output_dir) as staging:
        root = Path(staging)
        prepare_qc(root)
        tasks = []
        seen_splits = set()
        for record in manifest["sources"]:
            split = record["split"]
            pair = (
                record["source_id"],
                source_dir / record["image"],
                source_dir / record["mask"],
            )
            tasks.append(
                (
                    pair,
                    split,
                    np.asarray(record["starts"]),
                    root,
                    manifest["seed"],
                    split not in seen_splits,
                    record["background_options"],
                )
            )
            seen_splits.add(split)
        for index, record in enumerate(process_sources(tasks, workers), start=1):
            print(
                f"QC source {record['source_id']} complete ({index}/{len(tasks)})",
                flush=True,
            )
        for path in (root / "qc").glob("*/*.png"):
            destination = output_dir / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.replace(destination)


def generate_dataset(source_dir, output_dir, *, seed=2026, workers=4):
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if workers < 1 or seed < 0:
        raise ValueError("workers must be positive and seed nonnegative")
    pairs = source_pairs(source_dir)
    val_count = max(1, round(len(pairs) * 0.1))
    validation = set(np.random.default_rng(seed).permutation(len(pairs))[:val_count])
    plans = []
    total_bytes = 0
    seen_splits = set()
    for index, pair in enumerate(pairs):
        split = "validation" if index in validation else "train"
        chi, _ = load_pair(*pair[1:])
        starts = eligible_starts(chi)
        if not len(starts):
            raise ValueError(f"Source {pair[0]} has no eligible patches")
        preview = split not in seen_splits
        seen_splits.add(split)
        plans.append((pair, split, starts, preview))
        float_arrays = 2 + BACKGROUND_OPTIONS + (split == "train")
        total_bytes += len(starts) * PATCH_SIZE**3 * (1 + 4 * float_arrays)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_dir.parent).free < total_bytes:
        raise OSError(
            f"Insufficient free space for {total_bytes:,} bytes of patch arrays"
        )
    print(
        f"Generating {len(plans)} sources, {total_bytes / 2**30:.2f} GiB of patch arrays",
        flush=True,
    )
    with TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as staging:
        root = Path(staging) / "dataset"
        prepare_qc(root)
        tasks = [
            (pair, split, starts, root, seed, preview, None)
            for pair, split, starts, preview in plans
        ]
        records = []
        for record in process_sources(tasks, workers):
            records.append(record)
            print(
                f"Source {record['source_id']} complete ({len(records)}/{len(tasks)})",
                flush=True,
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "source_dir": str(source_dir),
            "seed": seed,
            "patch_size": PATCH_SIZE,
            "stride": STRIDE,
            "max_zeros": MAX_ZEROS,
            "padding_voxels": PADDING,
            "voxel_size_mm": [1, 1, 1],
            "b0": [0, 0, 1],
            "units": "ppm",
            "positive_scale": POSITIVE_SCALE,
            "background_options": BACKGROUND_OPTIONS,
            "exterior_baseline_range_ppm": [0, 1],
            "sphere_count": SPHERE_COUNT,
            "sphere_radius_voxels": SPHERE_RADII,
            "sphere_offset_magnitude_ppm": SPHERE_CONTRAST,
            "sphere_surface_clearance_voxels": SPHERE_CLEARANCE,
            "sources": records,
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, separators=(",", ":")) + "\n"
        )
        root.rename(output_dir)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--refresh-qc", action="store_true", help="Refresh PNGs in an existing dataset"
    )
    args = parser.parse_args()
    if args.refresh_qc:
        refresh_qc(args.output_dir, workers=args.workers)
        return
    generate_dataset(
        args.source_dir, args.output_dir, seed=args.seed, workers=args.workers
    )


if __name__ == "__main__":
    main()
