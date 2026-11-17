import numpy as np

from spinal_cord_qsm_benchmark.analysis import tissue_metrics
from spinal_cord_qsm_benchmark.statistics import (
    exact_sign_flip_wilcoxon,
    holm_adjust,
    paired_comparison,
    pairwise_agreement,
)


def test_tissue_metrics_keep_finite_zeros_and_report_coverage() -> None:
    values = np.array([0.0, 2.0, 4.0, np.nan], dtype=float)
    gm = np.array([1, 1, 0, 0], dtype=bool)
    wm = np.array([0, 0, 1, 1], dtype=bool)
    cord = np.ones(4, dtype=bool)
    result = tissue_metrics(values, gm, wm, cord)
    assert result["gm_mean"] == 1.0
    assert result["gm_sd"] == np.sqrt(2.0)
    assert result["gm_zero_fraction"] == 0.5
    assert result["wm_mean"] == 4.0
    assert result["wm_finite_fraction"] == 0.5
    assert result["gm_minus_wm"] == -3.0


def test_paired_statistics_have_opt_minus_def_sign_and_ci() -> None:
    result = paired_comparison(
        ["sub-01", "sub-02", "sub-03"],
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 4.0],
    )
    assert result["mean_difference_opt_minus_def"] == 4 / 3
    assert result["n"] == 3
    assert result["ci95_low"] < result["mean_difference_opt_minus_def"]
    assert result["ci95_high"] > result["mean_difference_opt_minus_def"]
    assert 0 <= result["wilcoxon_p"] <= 1


def test_exact_signed_rank_handles_ties_and_zero_differences() -> None:
    assert exact_sign_flip_wilcoxon(np.array([1.0, -1.0, 0.0])) == 1.0
    assert exact_sign_flip_wilcoxon(np.zeros(4)) == 1.0


def test_holm_adjustment_preserves_missing_values() -> None:
    adjusted = holm_adjust([0.01, 0.04, float("nan"), 0.03])
    assert adjusted[0] == 0.03
    assert adjusted[1] == 0.06
    assert np.isnan(adjusted[2])
    assert adjusted[3] == 0.06


def test_pairwise_map_agreement_uses_paired_finite_voxels() -> None:
    result = pairwise_agreement(
        np.array([1.0, 2.0, np.nan, 4.0]),
        np.array([1.0, 3.0, 8.0, 5.0]),
    )
    assert result["n_voxels"] == 3
    assert np.isclose(result["mae"], 2 / 3)
    assert np.isclose(result["rmse"], np.sqrt(2 / 3))
