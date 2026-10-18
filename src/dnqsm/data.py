"""Source caching and synthesis adapted from wh_net's maintained dataloader.

All source data are read-only. Synthesis returns the clean local field explicitly;
noise and background never enter that target. No local impulse augmentation.
"""

import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler, default_collate

from .physics import SPATIAL, continuous_dipole_kernel

TRAINING_CONDITIONS = {
    "snr_min": 40.0,
    "snr_max": 100.0,
    "snr_definition": "masked RMS(clean local field) / masked RMS(noise)",
    "exterior_source_amplitude_min": 0.01,
    "exterior_source_amplitude_max": 20.0,
    "reliability_min": 0.1,
}
CROP_RESIZE_SHAPE = (160, 160, 160)


def separable_dilation(mask):
    """Exactly the same binary dilation as max_pool3d(kernel_size=7)."""
    x = mask.float().unsqueeze(0)
    for kernel, padding in [
        ((7, 1, 1), (3, 0, 0)),
        ((1, 7, 1), (0, 3, 0)),
        ((1, 1, 7), (0, 0, 3)),
    ]:
        x = F.max_pool3d(x, kernel, stride=1, padding=padding)
    return x.squeeze(0).bool()


def training_records(data_dir, source_manifest=None):
    """Return explicit source identities; never silently split augmented files."""
    root = Path(data_dir)
    manifest_path = Path(source_manifest) if source_manifest else root / "subjects.json"
    identities = (
        json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    )
    if source_manifest and identities is None:
        raise FileNotFoundError(manifest_path)
    records = []
    for image in sorted(root.glob("image_*.pt")):
        index = int(image.stem.split("_")[-1])
        mask = root / f"mask_{index:04d}.pt"
        if not mask.is_file():
            raise FileNotFoundError(mask)
        if identities is not None:
            subject = identities[image.name]
        elif root.name in {"train_data2", "joint"} and 0 <= index < 651:
            # 0:315 mapping confirmed by the user; 315:651 from gen_data.py.
            subject = (
                f"qsm:{index % 105:04d}"
                if index < 315
                else f"separate:{(index - 315) % 56:04d}"
            )
        else:
            raise ValueError(
                "This dataset needs a JSON filename-to-subject source manifest"
            )
        records.append({"image": str(image), "mask": str(mask), "subject": subject})
    if not records:
        raise ValueError(f"No training volumes in {root}")
    return records


def sample_target_chi_rms(generator: torch.Generator) -> float:
    """Center physical-unit RMS near both references, with occasional strong cases."""
    tail = bool(torch.rand((), generator=generator) < 0.05)
    low, high = (0.04, 0.08) if tail else (0.018, 0.04)
    fraction = float(torch.rand((), generator=generator))
    return low * (high / low) ** fraction


class TrainingVolumes(Dataset):
    """Cache FP32 chi and boolean masks without modifying original data files."""

    def __init__(
        self,
        records,
        cache_dir,
        *,
        validation=False,
        views=1,
        seed=0,
        chi_scale_mode="legacy",
        crop_probability=0.0,
        crop_resize=True,
    ):
        self.records = records
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.validation, self.views, self.seed = validation, views, seed
        if not 0 <= crop_probability <= 1:
            raise ValueError("Crop probability must be between zero and one")
        self.crop_probability = 0.0 if validation else crop_probability
        self.crop_resize = crop_resize
        if chi_scale_mode not in ("legacy", "rms_mixture"):
            raise ValueError(f"Unknown chi scale mode: {chi_scale_mode}")
        self.chi_scale_mode = chi_scale_mode
        import multiprocessing

        self.epoch = multiprocessing.Value("q", 0)

    def __len__(self):
        return len(self.records) * self.views

    def set_epoch(self, epoch):
        self.epoch.value = epoch

    def _load(self, record):
        import hashlib

        paths = [Path(record[key]) for key in ("image", "mask")]
        signature = repr(
            [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
        )
        key = hashlib.sha256(signature.encode()).hexdigest()[:20]
        path = self.cache_dir / f"{paths[0].stem}_{key}.pt"
        if path.is_file():
            return torch.load(path, weights_only=True, mmap=True, map_location="cpu")
        chi = torch.load(paths[0], weights_only=True, map_location="cpu").float()
        mask = torch.load(paths[1], weights_only=True, map_location="cpu")
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError(f"Nonbinary source mask: {paths[1]}")
        mask = mask.bool()
        if chi.shape != mask.shape or chi.ndim != 4 or chi.shape[0] != 1:
            raise ValueError(f"Invalid image/mask shapes for {paths[0]}")
        if not bool(mask.any()) or not bool(torch.isfinite(chi).all()):
            raise ValueError(f"Empty mask or nonfinite source: {paths[0]}")
        result = {"chi": chi, "mask": mask, "dilated": separable_dilation(mask)}
        with NamedTemporaryFile(
            dir=self.cache_dir, suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            torch.save(result, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return result

    def __getitem__(self, index):
        # Training samplers may encode a repeated view above len(records).
        record = self.records[
            index // self.views if self.validation else index % len(self.records)
        ]
        epoch_stride = len(self) if self.chi_scale_mode == "legacy" else len(self) ** 2
        sample_seed = (
            self.seed
            + index
            + (0 if self.validation else self.epoch.value * epoch_stride)
        )
        generator = torch.Generator().manual_seed(sample_seed)
        loaded = self._load(record)
        shape = loaded["chi"].shape[-3:]
        # Axis swaps are valid here only for cubic, isotropic training volumes.
        permutation = (
            torch.randperm(3, generator=generator).tolist()
            if len(set(shape)) == 1
            else [0, 1, 2]
        )
        dims = (0, *(axis + 1 for axis in permutation))
        flips = tuple(
            axis for axis in (1, 2, 3) if torch.rand((), generator=generator) < 0.5
        )
        result = {}
        for key, volume in loaded.items():
            volume = volume.permute(dims)
            if flips:
                volume = volume.flip(flips)
            result[key] = volume.contiguous()
        if self.chi_scale_mode == "legacy":
            scale = float(torch.empty(()).uniform_(0.5, 3, generator=generator))
        else:
            chi_rms = (
                (result["chi"].square() * result["mask"]).sum() / result["mask"].sum()
            ).sqrt()
            scale = sample_target_chi_rms(generator) / max(float(chi_rms), 1e-8)
        result["chi"] = result["chi"] * scale
        original_shape = tuple(result["chi"].shape[-3:])
        crop_start = (0, 0, 0)
        crop_selected = bool(
            torch.rand(
                (), generator=torch.Generator().manual_seed(sample_seed + 73939133)
            )
            < self.crop_probability
        )
        if crop_selected:
            slices = []
            starts = []
            mask = result["mask"][0]
            for axis in range(3):
                occupied = mask.any(dim=tuple(i for i in range(3) if i != axis))
                indices = occupied.nonzero().flatten()
                start, end = int(indices[0]), int(indices[-1]) + 1
                if end - start < 2:
                    raise ValueError(
                        f"Crop has an axis shorter than two voxels: {record['image']}"
                    )
                slices.append(slice(start, end))
                starts.append(start)
            crop_start = tuple(starts)
            for key in ("chi", "mask", "dilated"):
                result[key] = result[key][(slice(None), *slices)].contiguous()
        crop_shape = tuple(result["chi"].shape[-3:])
        resize_applied = (
            crop_selected and self.crop_resize and crop_shape != CROP_RESIZE_SHAPE
        )
        if resize_applied:
            result["chi"] = F.interpolate(
                result["chi"].unsqueeze(0),
                size=CROP_RESIZE_SHAPE,
                mode="trilinear",
                align_corners=False,
            )[0]
            result["mask"] = F.interpolate(
                result["mask"].unsqueeze(0).float(),
                size=CROP_RESIZE_SHAPE,
                mode="nearest",
            )[0].bool()
            result["chi"] *= result["mask"]
            # Keep the exterior-source exclusion at seven output-grid voxels.
            result["dilated"] = separable_dilation(result["mask"])
        result.update(
            original_shape=torch.tensor(original_shape),
            crop_start=torch.tensor(crop_start),
            crop_shape=torch.tensor(crop_shape),
            crop_applied=crop_selected,
            resize_applied=resize_applied,
        )
        result.update(seed=sample_seed, subject=record["subject"])
        return result


def collate_shapes(samples):
    """Stack only identical shapes; never reintroduce spatial zero padding."""
    groups = defaultdict(list)
    for sample in samples:
        groups[tuple(sample["chi"].shape)].append(sample)
    return [default_collate(group) for group in groups.values()]


def shape_groups(batch):
    return batch if isinstance(batch, list) else [batch]


class ShapeFieldSynthesizer:
    """Bounded device-kernel cache for native, potentially rectangular grids."""

    def __init__(self, device, capacity=4):
        self.device, self.capacity = device, capacity
        self.cache = OrderedDict()

    def __call__(self, batch, **kwargs):
        shape = tuple(batch["chi"].shape[-3:])
        if shape not in self.cache:
            if len(self.cache) == self.capacity:
                self.cache.popitem(last=False)
            self.cache[shape] = FieldSynthesizer(shape, self.device)
        self.cache.move_to_end(shape)
        return self.cache[shape](batch, **kwargs)


class FieldSynthesizer(torch.nn.Module):
    """Real-FFT dipole simulation, diverse exterior sources and RMS-defined SNR."""

    def __init__(self, shape, device):
        super().__init__()
        kernel = continuous_dipole_kernel(tuple(shape))
        self.register_buffer(
            "dipole",
            torch.from_numpy(kernel[..., : shape[-1] // 2 + 1].copy())
            .float()[None, None]
            .to(device),
        )
        self.shape = tuple(shape)

    @torch.no_grad()
    def forward(
        self,
        batch,
        *,
        background_mode: str = "legacy",
    ):
        if background_mode not in ("legacy", "ratio"):
            raise ValueError(f"Unknown background mode: {background_mode}")
        device = self.dipole.device
        chi = batch["chi"].to(device, non_blocking=True).float()
        mask = batch["mask"].to(device, non_blocking=True).float()
        dilated = batch["dilated"].to(device, non_blocking=True)
        chi = chi * mask
        source_fields, noises, snrs, ratios = [], [], [], []
        for index, seed in enumerate(batch["seed"].tolist()):
            # Independent per-sample streams make validation invariant to batch size.
            generator = torch.Generator(device=device).manual_seed(seed + 1000003)

            def rand(*shape):
                return torch.rand(shape, device=device, generator=generator)

            smooth = F.interpolate(
                rand(1, 1, 5, 5, 5),
                size=self.shape,
                mode="trilinear",
                align_corners=False,
            )[0]
            selector = rand()
            uniform = rand() < 0.5
            exterior = torch.where(uniform, torch.ones_like(smooth), 2.0 * smooth - 1.0)
            amplitude = (
                TRAINING_CONDITIONS["exterior_source_amplitude_min"]
                + (
                    TRAINING_CONDITIONS["exterior_source_amplitude_max"]
                    - TRAINING_CONDITIONS["exterior_source_amplitude_min"]
                )
                * rand()
            )
            if background_mode == "ratio":
                ratios.append(0.03 * (3.0 / 0.03) ** float(rand()))
            source_fields.append(exterior * (~dilated[index]) * amplitude)
            signal = chi[index]
            low = signal.masked_fill(mask[index] == 0, torch.inf).amin()
            high = signal.masked_fill(mask[index] == 0, -torch.inf).amax()
            contrast = ((signal - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
            reliability = torch.where(
                selector < 0.5,
                torch.ones_like(smooth),
                torch.where(selector < 0.75, smooth, contrast),
            )
            reliability_floor = TRAINING_CONDITIONS["reliability_min"]
            reliability = (
                reliability_floor + (1.0 - reliability_floor) * reliability
            ) * mask[index]
            noises.append(
                torch.randn(signal.shape, device=device, generator=generator)
                / reliability.clamp_min(reliability_floor).sqrt()
            )
            snrs.append(
                TRAINING_CONDITIONS["snr_min"]
                + (TRAINING_CONDITIONS["snr_max"] - TRAINING_CONDITIONS["snr_min"])
                * rand()
            )

        def inverse(x):
            return torch.fft.irfftn(x, s=self.shape, dim=SPATIAL)

        local = inverse(self.dipole * torch.fft.rfftn(chi, dim=SPATIAL)) * mask
        background = (
            inverse(
                self.dipole * torch.fft.rfftn(torch.stack(source_fields), dim=SPATIAL)
            )
            * mask
        )
        count = mask.sum(dim=SPATIAL, keepdim=True).clamp_min(1)

        def rms(x):
            return (
                ((x.square() * mask).sum(dim=SPATIAL, keepdim=True) / count)
                .sqrt()
                .clamp_min(1e-8)
            )

        if background_mode == "ratio":
            target = torch.tensor(ratios, device=device).reshape(-1, 1, 1, 1, 1)
            background = background * rms(local) / (rms(background) * target)

        noise = torch.stack(noises) * mask
        noise = (
            noise
            * rms(local)
            / (rms(noise) * torch.stack(snrs).reshape(-1, 1, 1, 1, 1))
        )
        phase = (local + background + noise) * mask
        return dict(
            phase=phase,
            mask=mask,
            dipole=self.dipole,
            chi=chi,
            background=background,
            local=local,
            noise=noise,
        )


def make_subject_split(records, fraction, seed):
    cohorts = defaultdict(set)
    for record in records:
        cohort = record["subject"].split(":")[0]
        if cohort != "synthetic":
            cohorts[cohort].add(record["subject"])
    generator = torch.Generator().manual_seed(seed)
    validation_subjects = set()
    for subjects in cohorts.values():
        subjects = sorted(subjects)
        if len(subjects) < 2:
            raise ValueError("Each source cohort needs at least two subjects")
        count = max(1, min(len(subjects) - 1, round(len(subjects) * fraction)))
        validation_subjects.update(
            subjects[i]
            for i in torch.randperm(len(subjects), generator=generator)[:count].tolist()
        )
    train = [r for r in records if r["subject"] not in validation_subjects]
    val = [r for r in records if r["subject"] in validation_subjects]
    assert not {r["subject"] for r in train} & {r["subject"] for r in val}
    return train, val


class BalancedSourceSampler(Sampler[int]):
    """Draw a fixed source mix with fresh deterministic views for repeated records."""

    def __init__(self, records: list[dict], synthetic_fraction: float, seed: int):
        self.original = [
            index
            for index, row in enumerate(records)
            if not row["subject"].startswith("synthetic:")
        ]
        self.synthetic = [
            index
            for index, row in enumerate(records)
            if row["subject"].startswith("synthetic:")
        ]
        self.size = len(records)
        self.synthetic_fraction = synthetic_fraction
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return self.size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        n_synthetic = round(self.size * self.synthetic_fraction)
        if not self.synthetic:
            n_synthetic = 0
        if not self.original:
            n_synthetic = self.size

        def draw(pool, count, cohort):
            selected = []
            offset = self.epoch * count
            while len(selected) < count:
                cycle, within = divmod(offset, len(pool))
                cycle_generator = torch.Generator().manual_seed(
                    self.seed + cohort * 1000003 + cycle
                )
                order = torch.randperm(len(pool), generator=cycle_generator).tolist()
                take = min(count - len(selected), len(pool) - within)
                selected.extend(pool[index] for index in order[within : within + take])
                offset += take
            return selected

        selected = draw(self.original, self.size - n_synthetic, 0) + draw(
            self.synthetic, n_synthetic, 1
        )
        occurrences = defaultdict(int)
        encoded = []
        for index in selected:
            encoded.append(index + occurrences[index] * self.size)
            occurrences[index] += 1
        order = torch.randperm(len(encoded), generator=generator).tolist()
        return iter(encoded[index] for index in order)
