import json
from pathlib import Path

import torch


def make_dataset(directory: Path, shape=(12, 12, 12)) -> Path:
    root = directory / "sources"
    root.mkdir()
    generator = torch.Generator().manual_seed(312)
    identities = {}
    for index, subject in enumerate(
        (
            "qsm:a",
            "qsm:a",
            "qsm:b",
            "qsm:c",
            "separate:a",
            "separate:b",
            "synthetic:a",
            "synthetic:b",
        )
    ):
        mask = torch.zeros(1, *shape, dtype=torch.bool)
        mask[:, 4:-4, 4:-4, 4:-4] = True
        chi = torch.randn(1, *shape, generator=generator) * 0.03 * mask
        name = f"image_{index:04d}.pt"
        torch.save(chi, root / name)
        torch.save(mask, root / f"mask_{index:04d}.pt")
        identities[name] = subject
    (root / "subjects.json").write_text(json.dumps(identities))
    return root
