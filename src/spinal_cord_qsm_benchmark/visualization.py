"""Publication-oriented figures for map, ROI, and algorithm comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd


COLORS = {"def": "#336699", "opt": "#d35f4b"}
MEASURE_LABELS = {
    "gm_mean": "Gray matter mean",
    "wm_mean": "White matter mean",
    "gm_minus_wm": "GM - WM contrast",
}


def _canonical_sagittal_slice(
    values: np.ndarray, image: nib.spatialimages.SpatialImage
) -> np.ndarray:
    canonical = nib.as_closest_canonical(nib.Nifti1Image(values, image.affine))
    data = canonical.get_fdata(dtype=np.float32)
    return data[data.shape[0] // 2, :, :].T


def _save_roi_figure(roi: pd.DataFrame, kind: str, output_dir: Path) -> Path | None:
    if roi.empty or "kind" not in roi.columns:
        return None
    subset = roi.loc[roi["kind"] == kind]
    if subset.empty:
        return None

    methods = sorted(subset["method"].dropna().unique())
    configurations = list(subset["configuration"].dropna().unique())
    configurations.sort(key=lambda name: (name != "def", name))
    figure, axes = plt.subplots(3, 1, figsize=(max(9, len(methods) * 1.5), 10))
    participant_ids = sorted(subset["participant"].unique())

    for axis, measure in zip(axes, MEASURE_LABELS, strict=True):
        finite = subset.loc[np.isfinite(subset[measure].astype(float))]
        for method_index, method in enumerate(methods):
            method_data = finite.loc[finite["method"] == method]
            config_positions: dict[str, float] = {}
            for config_index, configuration in enumerate(configurations):
                position = method_index + (config_index - (len(configurations) - 1) / 2) * 0.25
                config_positions[configuration] = position
                values = method_data.loc[
                    method_data["configuration"] == configuration
                ]
                xs = [
                    position
                    + (participant_ids.index(participant) - (len(participant_ids) - 1) / 2)
                    * 0.012
                    for participant in values["participant"]
                ]
                axis.scatter(
                    xs,
                    values[measure],
                    s=20,
                    alpha=0.75,
                    color=COLORS.get(configuration, "#777777"),
                    label=configuration if method_index == 0 else None,
                    zorder=3,
                )
                if not values.empty:
                    axis.scatter(
                        [position],
                        [values[measure].mean()],
                        marker="D",
                        s=38,
                        color=COLORS.get(configuration, "#777777"),
                        edgecolor="white",
                        linewidth=0.5,
                        zorder=4,
                    )

            if len(config_positions) == 2:
                first, second = configurations
                paired = method_data.pivot(
                    index="participant", columns="configuration", values=measure
                )
                if first in paired and second in paired:
                    for participant in paired.dropna(subset=[first, second]).index:
                        axis.plot(
                            [config_positions[first], config_positions[second]],
                            [paired.loc[participant, first], paired.loc[participant, second]],
                            color="#9b9b9b",
                            linewidth=0.7,
                            alpha=0.35,
                            zorder=1,
                        )

        axis.set_ylabel(f"{MEASURE_LABELS[measure]}\n(source units)")
        axis.set_xticks(range(len(methods)), methods)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(title="Configuration", frameon=False, loc="best")
    axes[-1].set_xlabel("Dipole-inversion method" if kind == "chimap" else "Background-field-removal method")
    figure.suptitle(f"Participant-level ROI estimates: {kind}", y=1.01)
    figure.tight_layout()
    path = output_dir / f"roi_metrics_{kind}.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def _save_agreement_figure(agreement: pd.DataFrame, output_dir: Path) -> Path | None:
    if agreement.empty:
        return None
    kinds = [kind for kind in ("chimap", "localfield") if kind in agreement["kind"].unique()]
    figure, axes = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 5), squeeze=False)
    configurations = sorted(
        agreement["configuration"].dropna().unique(),
        key=lambda name: (name != "def", name),
    )
    for axis, kind in zip(axes[0], kinds, strict=True):
        subset = agreement.loc[agreement["kind"] == kind]
        data = [
            subset.loc[subset["configuration"] == configuration, "mae"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy()
            for configuration in configurations
        ]
        if any(len(values) for values in data):
            boxes = axis.boxplot(data, labels=configurations, patch_artist=True)
            for box, configuration in zip(boxes["boxes"], configurations, strict=True):
                box.set_facecolor(COLORS.get(configuration, "#777777"))
                box.set_alpha(0.7)
        axis.set_title(kind)
        axis.set_ylabel("Pairwise map MAE (source units)")
        axis.set_xlabel("Configuration")
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Inter-algorithm map disagreement within the GM/WM ROI union")
    figure.tight_layout()
    path = output_dir / "algorithm_agreement.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def _save_map_comparison(
    example: dict[str, Any] | None, output_dir: Path
) -> Path | None:
    if example is None or "opt_values" not in example:
        return None
    image = example["image"]
    evaluation_mask = example["evaluation_mask"]
    default_values = _canonical_sagittal_slice(example["values"], image)
    optimized_values = _canonical_sagittal_slice(example["opt_values"], image)
    canonical_mask = (
        _canonical_sagittal_slice(evaluation_mask.astype(np.float32), image) > 0
    )
    default_values = np.where(canonical_mask, default_values, np.nan)
    optimized_values = np.where(canonical_mask, optimized_values, np.nan)
    difference = optimized_values - default_values
    in_cord = np.abs(difference[np.isfinite(difference)])
    if not in_cord.size:
        return None
    limit = float(np.percentile(in_cord, 98))
    if not np.isfinite(limit) or limit == 0:
        limit = float(np.max(in_cord)) or 1.0
    extent = np.nanpercentile(
        np.concatenate(
            [
                np.abs(default_values[np.isfinite(default_values)]),
                np.abs(optimized_values[np.isfinite(optimized_values)]),
            ]
        ),
        98,
    )
    if not np.isfinite(extent) or extent == 0:
        extent = 1.0

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, values, title in (
        (axes[0], default_values, "Brain-recommended (def)"),
        (axes[1], optimized_values, "Spinal-cord-optimized (opt)"),
    ):
        image_artist = axis.imshow(
            values, cmap="coolwarm", vmin=-extent, vmax=extent, origin="lower"
        )
        axis.set_title(title)
        axis.set_xlabel("Anterior - posterior voxel")
        axis.set_ylabel("Inferior - superior voxel")
        figure.colorbar(image_artist, ax=axis, shrink=0.75, label="Source units")

    difference_artist = axes[2].imshow(
        difference,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        origin="lower",
    )
    axes[2].set_title("opt - def")
    axes[2].set_xlabel("Anterior - posterior voxel")
    axes[2].set_ylabel("Inferior - superior voxel")
    figure.colorbar(
        difference_artist, ax=axes[2], shrink=0.75, label="Source units"
    )
    figure.suptitle(
        f"Susceptibility-map comparison: {example['participant']}, {example['method']}"
    )
    figure.tight_layout()
    path = output_dir / "susceptibility_map_comparison.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def create_figures(results: dict[str, Any], output_dir: Path) -> list[Path]:
    """Render all figures supported by the available data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for kind in ("chimap", "localfield"):
        figure_path = _save_roi_figure(results["roi_metrics"], kind, output_dir)
        if figure_path:
            paths.append(figure_path)
    agreement_path = _save_agreement_figure(results["map_agreement"], output_dir)
    if agreement_path:
        paths.append(agreement_path)
    map_path = _save_map_comparison(results["map_example"], output_dir)
    if map_path:
        paths.append(map_path)
    return paths
