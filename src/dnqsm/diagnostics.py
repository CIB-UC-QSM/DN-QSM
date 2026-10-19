"""Detached quality summaries and occasional per-objective gradient telemetry."""

import itertools

import torch

from .physics import apply_dipole, masked_mean


def loss_gradient_metrics(model, terms, weights):
    """Probe the existing graph without updating .grad, weights, or RNG state."""
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    indices = {
        branch: [
            i for i, (name, _) in enumerate(named) if name.startswith(branch + ".")
        ]
        for branch in ("filter", "null_network")
    }
    vectors = {}
    metrics = {}
    for name, term in terms.items():
        gradients = torch.autograd.grad(
            term,
            [parameter for _, parameter in named],
            retain_graph=True,
            allow_unused=True,
        )
        vectors[name] = {}
        for branch, positions in indices.items():
            # Keep cross-objective vectors on the host. The large overfit
            # profile must not retain five full parameter gradients in VRAM.
            vector = (
                torch.cat(
                    [
                        (
                            gradients[i]
                            if gradients[i] is not None
                            else torch.zeros_like(named[i][1])
                        )
                        .detach()
                        .flatten()
                        for i in positions
                    ]
                )
                if positions
                else torch.empty(0)
            ).cpu()
            vectors[name][branch] = vector
            norm = float(vector.norm())
            metrics[f"grad_{branch}_{name}"] = norm
            metrics[f"weighted_grad_{branch}_{name}"] = norm * getattr(weights, name)
        del gradients
    for first, second in itertools.combinations(vectors, 2):
        for branch in indices:
            a, b = vectors[first][branch], vectors[second][branch]
            denominator = a.norm() * b.norm()
            if float(denominator) > 0:
                metrics[f"grad_cosine_{branch}_{first}_{second}"] = float(
                    torch.dot(a, b) / denominator
                )
    return metrics


@torch.no_grad()
def quality_totals(prediction, data, *, predicted_field=None):
    """Additive sums/counts stay correct across ragged groups and accumulation."""
    mask = data["mask"]
    if isinstance(prediction, torch.Tensor):
        comparisons = (("local", prediction, data["local"]),)
    else:
        field = (
            apply_dipole(prediction.chi, data["dipole"])
            if predicted_field is None
            else predicted_field.detach()
        )
        comparisons = (
            ("chi", prediction.chi, data["chi"]),
            ("tkd", prediction.tkd, data["chi"]),
            ("local", prediction.local_field, data["local"]),
            ("physics", field, data["local"]),
        )
    local_rms = masked_mean(data["local"].square(), mask).sqrt().flatten()
    background_rms = masked_mean(data["background"].square(), mask).sqrt().flatten()
    ratio = local_rms / background_rms.clamp_min(1e-12)
    clean = background_rms == 0
    buckets = {
        "clean": clean,
        "strong": (~clean) & (ratio < 0.03),
        "typical": (~clean) & (ratio >= 0.03) & (ratio <= 3),
        "weak": (~clean) & (ratio > 3),
    }
    totals = {
        f"bucket_{name}_samples": int(selected.sum())
        for name, selected in buckets.items()
    }
    for name, value, reference in comparisons:
        denominator = masked_mean(reference.square(), mask).sqrt().flatten()
        valid = denominator > 0
        errors = (
            100
            * masked_mean((value - reference).square(), mask).sqrt().flatten()
            / denominator.clamp_min(1e-20)
        )
        selections = {name + "_nrmse": valid}
        selections.update(
            {
                f"bucket_{bucket}_{name}_nrmse": selected & valid
                for bucket, selected in buckets.items()
            }
        )
        for key, selected in selections.items():
            totals[f"quality_{key}_sum"] = float(errors[selected].sum())
            totals[f"quality_{key}_count"] = int(selected.sum())
    return totals


def summarize_quality(totals):
    metrics = {
        key: value for key, value in totals.items() if not key.startswith("quality_")
    }
    for key, value in totals.items():
        if key.startswith("quality_") and key.endswith("_sum"):
            count = totals[key.removesuffix("_sum") + "_count"]
            if count:
                metrics[key.removeprefix("quality_").removesuffix("_sum")] = (
                    value / count
                )
    return metrics
