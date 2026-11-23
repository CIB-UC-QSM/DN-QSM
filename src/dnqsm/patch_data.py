"""Read materialized AHEAD QSM patches for PyTorch training."""

import hashlib
import json
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from .physics import dipole_tensor

PATCH_SIZE = 64
BACKGROUND_OPTIONS = 4
VARIANTS = ("original", "scaled", "inverted", "scaled_inverted")
SCHEMA_VERSION = 1


def scale_positive(chi, scale):
    return np.where(chi > 0, chi * np.float32(scale), chi)


class QSMPatchDataset(Dataset):
    """Randomize the stored background on every read, including validation."""

    def __init__(self, root, split="train", *, seed=2026, fixed_backgrounds=False):
        if split not in ("train", "validation"):
            raise ValueError("split must be train or validation")
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported QSM patch dataset version")
        self.records = [r for r in self.manifest["sources"] if r["split"] == split]
        if not self.records:
            raise ValueError(f"No patches in split {split}")
        self.variants = VARIANTS if split == "train" else VARIANTS[:1]
        self.ends = np.cumsum(
            [len(r["starts"]) * len(self.variants) for r in self.records]
        )
        self.seed = seed
        self.fixed_backgrounds = fixed_backgrounds
        self._rng = None
        self._worker_seed = None
        self._cache = OrderedDict()

    def __len__(self):
        return int(self.ends[-1])

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_cache=OrderedDict(), _rng=None)
        return state

    def dipole(self, device="cpu"):
        """Return the DN-QSM multiplier on a 64³ patch grid."""
        return dipole_tensor((PATCH_SIZE,) * 3, device=device)

    def _arrays(self, record):
        source = record["source_id"]
        if source not in self._cache:
            folder = self.root / "patches" / source
            names = ["chi", "mask", "local", "background"]
            if record["split"] == "train":
                names.append("local_scaled")
            self._cache[source] = {
                name: np.load(folder / f"{name}.npy", mmap_mode="r") for name in names
            }
            if len(self._cache) > 8:
                self._cache.popitem(last=False)
        self._cache.move_to_end(source)
        return self._cache[source]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        source_index = bisect_right(self.ends, index)
        offset = int(self.ends[source_index - 1]) if source_index else 0
        patch_index, variant_index = divmod(index - offset, len(self.variants))
        record = self.records[source_index]
        arrays = self._arrays(record)
        if self.fixed_backgrounds:
            key = f"{self.seed}:{record['source_id']}:{patch_index}".encode()
            option = (
                hashlib.blake2b(key, digest_size=1).digest()[0] % BACKGROUND_OPTIONS
            )
        else:
            worker = get_worker_info()
            seed = self.seed if worker is None else worker.seed
            worker_key = (worker is not None, seed)
            if self._rng is None or self._worker_seed != worker_key:
                self._rng = np.random.default_rng(seed)
                self._worker_seed = worker_key
            option = int(self._rng.integers(BACKGROUND_OPTIONS))

        chi = arrays["chi"][patch_index].copy()
        scaled = variant_index in (1, 3)
        if scaled:
            chi = scale_positive(chi, self.manifest["positive_scale"])
        local = arrays["local_scaled" if scaled else "local"][patch_index].copy()
        if variant_index >= 2:
            chi *= -1
            local *= -1
        background = arrays["background"][patch_index, option].copy()
        return {
            "chi": torch.from_numpy(chi[None]),
            "mask": torch.from_numpy(arrays["mask"][patch_index][None].copy()),
            "local_field": torch.from_numpy(local[None]),
            "background_field": torch.from_numpy(background[None]),
            "total_field": torch.from_numpy((local + background)[None]),
            "background_option": option,
            "source_id": record["source_id"],
            "variant": self.variants[variant_index],
            "patch_start": torch.tensor(record["starts"][patch_index]),
        }


def make_patch_dataloader(
    root,
    split="train",
    *,
    batch_size=1,
    num_workers=0,
    seed=2026,
    fixed_backgrounds=False,
    **kwargs,
):
    dataset = QSMPatchDataset(
        root, split, seed=seed, fixed_backgrounds=fixed_backgrounds
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=split == "train",
        generator=torch.Generator().manual_seed(seed),
        **kwargs,
    )
