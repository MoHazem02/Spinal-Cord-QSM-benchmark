"""Paired, participant-level summaries for small samples."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import stats


def paired_comparison(
    participants: list[str],
    brain_recommended: list[float],
    spinal_optimized: list[float],
) -> dict[str, float | int]:
    """Summarize paired opt-minus-def differences and small-sample tests."""
    pairs = [
        (participant, float(default), float(optimized))
        for participant, default, optimized in zip(
            participants, brain_recommended, spinal_optimized, strict=True
        )
        if np.isfinite(default) and np.isfinite(optimized)
    ]
    differences = np.asarray(
        [optimized - default for _, default, optimized in pairs], dtype=float
    )
    n = len(differences)
    result: dict[str, float | int] = {
        "n": n,
        "mean_def": float(np.mean([item[1] for item in pairs])) if n else np.nan,
        "mean_opt": float(np.mean([item[2] for item in pairs])) if n else np.nan,
        "mean_difference_opt_minus_def": float(np.mean(differences)) if n else np.nan,
        "sd_difference": float(np.std(differences, ddof=1)) if n > 1 else np.nan,
        "ci95_low": np.nan,
        "ci95_high": np.nan,
        "cohen_dz": np.nan,
        "paired_t_p": np.nan,
        "wilcoxon_p": np.nan,
    }
    if n == 0:
        return result
    if n > 1:
        sd = float(np.std(differences, ddof=1))
        if sd == 0:
            result["ci95_low"] = result["mean_difference_opt_minus_def"]
            result["ci95_high"] = result["mean_difference_opt_minus_def"]
            result["cohen_dz"] = np.nan
            result["paired_t_p"] = 1.0 if np.mean(differences) == 0 else 0.0
        else:
            sem = sd / np.sqrt(n)
            margin = float(stats.t.ppf(0.975, df=n - 1) * sem)
            mean_difference = float(np.mean(differences))
            result["ci95_low"] = mean_difference - margin
            result["ci95_high"] = mean_difference + margin
            result["cohen_dz"] = mean_difference / sd
            result["paired_t_p"] = float(
                stats.ttest_1samp(differences, popmean=0).pvalue
            )
        result["wilcoxon_p"] = exact_sign_flip_wilcoxon(differences)
    return result


def exact_sign_flip_wilcoxon(differences: np.ndarray) -> float:
    """Compute an exact paired signed-rank p-value, including tied absolute ranks."""
    nonzero = np.asarray(differences, dtype=float)
    nonzero = nonzero[np.isfinite(nonzero) & (nonzero != 0)]
    if nonzero.size == 0:
        return 1.0

    ranks = stats.rankdata(np.abs(nonzero), method="average")
    observed = float(np.sum(ranks[nonzero > 0]))
    center = float(np.sum(ranks) / 2)
    observed_distance = abs(observed - center)
    assignment_count = 1 << nonzero.size
    at_least_as_extreme = 0
    for signs in range(assignment_count):
        positive = np.fromiter(
            (bool(signs & (1 << index)) for index in range(nonzero.size)),
            dtype=bool,
            count=nonzero.size,
        )
        statistic = float(np.sum(ranks[positive]))
        if abs(statistic - center) >= observed_distance - 1e-12:
            at_least_as_extreme += 1
    return at_least_as_extreme / assignment_count


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, preserving missing entries."""
    adjusted = [float("nan")] * len(p_values)
    valid = [(index, value) for index, value in enumerate(p_values) if np.isfinite(value)]
    ordered = sorted(valid, key=lambda item: item[1])
    running_max = 0.0
    count = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        running_max = max(running_max, (count - rank) * value)
        adjusted[index] = min(1.0, running_max)
    return adjusted


def pairwise_agreement(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    """Return correlation and absolute/squared map disagreement for paired voxels."""
    valid = np.isfinite(first) & np.isfinite(second)
    a = np.asarray(first[valid], dtype=float)
    b = np.asarray(second[valid], dtype=float)
    count = int(a.size)
    if count == 0:
        return {"n_voxels": 0, "pearson_r": np.nan, "mae": np.nan, "rmse": np.nan}

    correlation = (
        float(np.corrcoef(a, b)[0, 1])
        if count > 1 and np.std(a) > 0 and np.std(b) > 0
        else np.nan
    )
    delta = a - b
    return {
        "n_voxels": count,
        "pearson_r": correlation,
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
    }


def pairwise_indices(count: int) -> list[tuple[int, int]]:
    """Return unique unordered pairs for a sequence of algorithms."""
    return list(combinations(range(count), 2))
