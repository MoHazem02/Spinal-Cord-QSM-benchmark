"""Core analysis of published QSM maps and available anatomical masks."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

from .dataset import (
    ImageGeometryError,
    ImageRecord,
    assert_same_grid,
    discover_dataset,
    inspect_images,
)
from .statistics import holm_adjust, paired_comparison, pairwise_agreement, pairwise_indices


def _read_mask(
    records: list[ImageRecord],
    qc_by_path: dict[str, dict[str, object]],
    kind: str,
) -> tuple[np.ndarray | None, nib.spatialimages.SpatialImage | None]:
    record = next((item for item in records if item.kind == kind), None)
    if record is None or qc_by_path.get(str(record.path), {}).get("status") != "ok":
        return None, None
    image = nib.load(str(record.path))
    data = image.get_fdata(dtype=np.float32, caching="unchanged")
    return np.isfinite(data) & (data > 0), image


def tissue_metrics(
    values: np.ndarray,
    gray_mask: np.ndarray,
    white_mask: np.ndarray,
    analysis_mask: np.ndarray,
) -> dict[str, float | int]:
    """Summarize finite voxel values in the provided anatomical ROI masks."""
    if not (
        values.shape
        == gray_mask.shape
        == white_mask.shape
        == analysis_mask.shape
    ):
        raise ValueError("Map and mask dimensions must match before ROI analysis")

    gray_roi = gray_mask & analysis_mask
    white_roi = white_mask & analysis_mask
    gray_values = values[gray_roi & np.isfinite(values)]
    white_values = values[white_roi & np.isfinite(values)]
    result: dict[str, float | int] = {}

    for tissue, sample, roi in (
        ("gm", gray_values, gray_roi),
        ("wm", white_values, white_roi),
    ):
        result[f"{tissue}_n_mask"] = int(np.count_nonzero(roi))
        result[f"{tissue}_n_finite"] = int(sample.size)
        result[f"{tissue}_finite_fraction"] = (
            float(sample.size / np.count_nonzero(roi))
            if np.count_nonzero(roi)
            else np.nan
        )
        result[f"{tissue}_zero_fraction"] = (
            float(np.count_nonzero(sample == 0) / sample.size) if sample.size else np.nan
        )
        result[f"{tissue}_mean"] = float(np.mean(sample)) if sample.size else np.nan
        result[f"{tissue}_sd"] = (
            float(np.std(sample, ddof=1)) if sample.size > 1 else np.nan
        )

    result["gm_wm_overlap_voxels"] = int(np.count_nonzero(gray_roi & white_roi))
    result["gm_minus_wm"] = (
        float(result["gm_mean"] - result["wm_mean"])
        if np.isfinite(result["gm_mean"]) and np.isfinite(result["wm_mean"])
        else np.nan
    )
    return result


def _expected_map_records(records: list[ImageRecord]) -> list[ImageRecord]:
    return [record for record in records if record.kind in ("chimap", "localfield")]


def _sidecar_units(path: Path) -> str:
    sidecar = path.with_suffix("").with_suffix(".json")
    if not sidecar.is_file():
        return "not specified"
    import json

    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    return str(metadata.get("Units", metadata.get("units", "not specified")))


def run_analysis(
    data_root: Path,
    output_dir: Path,
    participants: list[str],
    algorithms: dict[str, list[str]],
    configurations: list[str],
) -> dict[str, Any]:
    """Run discovery, spatial QC, ROI analysis, map agreement, and paired comparisons."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, inventory_rows = discover_dataset(
        data_root, participants, algorithms, configurations
    )
    if len(configurations) != 2 or set(configurations) != {"def", "opt"}:
        raise ValueError("configurations must contain exactly 'def' and 'opt'")
    qc_rows = inspect_images(records)
    pd.DataFrame(inventory_rows).to_csv(output_dir / "inventory.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(output_dir / "image_qc.csv", index=False)
    qc_by_path = {str(row["path"]): row for row in qc_rows}

    records_by_participant: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        records_by_participant[record.participant].append(record)

    roi_rows: list[dict[str, object]] = []
    agreement_rows: list[dict[str, object]] = []
    dispersion_rows: list[dict[str, object]] = []
    map_example: dict[str, Any] | None = None
    valid_map_count = 0
    analysis_issues: list[dict[str, str]] = []

    for participant in participants:
        participant_records = records_by_participant[participant]
        gray_mask, gray_image = _read_mask(participant_records, qc_by_path, "gm_mask")
        white_mask, white_image = _read_mask(participant_records, qc_by_path, "wm_mask")

        if gray_mask is None or white_mask is None:
            missing = [
                name
                for name, mask in (
                    ("gm_mask", gray_mask),
                    ("wm_mask", white_mask),
                )
                if mask is None
            ]
            analysis_issues.append(
                {
                    "participant": participant,
                    "issue": f"Required anatomical masks unavailable or invalid: {', '.join(missing)}",
                }
            )
            continue

        if gray_mask.shape != white_mask.shape or gray_image is None or white_image is None:
            analysis_issues.append(
                {
                    "participant": participant,
                    "issue": "Gray- and white-matter mask dimensions differ",
                }
            )
            continue

        analysis_mask = gray_mask | white_mask
        cord_qc = next(
            (
                row
                for row in qc_by_path.values()
                if row["participant"] == participant and row["kind"] == "cord_mask"
            ),
            None,
        )
        if cord_qc is None or cord_qc["status"] != "ok":
            detail = (
                f": {cord_qc['issue']}"
                if cord_qc is not None and cord_qc["issue"]
                else ""
            )
            analysis_issues.append(
                {
                    "participant": participant,
                    "issue": (
                        "Cord mask unavailable as a compatible 3D image; "
                        "map comparisons use the union of the published GM and WM masks"
                        f"{detail}"
                    ),
                }
            )

        participant_maps: dict[
            tuple[str, str],
            list[tuple[str, np.ndarray, nib.spatialimages.SpatialImage]],
        ] = defaultdict(list)
        for record in _expected_map_records(participant_records):
            qc = qc_by_path.get(str(record.path), {})
            if qc.get("status") != "ok":
                continue
            try:
                image = nib.load(str(record.path))
                assert_same_grid(gray_image, image)
                data = image.get_fdata(dtype=np.float32, caching="unchanged")
            except (
                OSError,
                ValueError,
                EOFError,
                nib.filebasedimages.ImageFileError,
                ImageGeometryError,
            ) as error:
                analysis_issues.append(
                    {"participant": participant, "issue": f"{record.path}: {error}"}
                )
                continue

            valid_map_count += 1
            key = (record.kind, str(record.configuration))
            participant_maps[key].append((record.method or "", data, image))
            metrics = tissue_metrics(data, gray_mask, white_mask, analysis_mask)
            roi_rows.append(
                {
                    "participant": participant,
                    "kind": record.kind,
                    "method": record.method,
                    "configuration": record.configuration,
                    "units": _sidecar_units(record.path),
                    **metrics,
                }
            )
            if (
                map_example is None
                and record.kind == "chimap"
                and record.configuration == configurations[0]
            ):
                map_example = {
                    "participant": participant,
                    "method": record.method,
                    "configuration": record.configuration,
                    "image": image,
                    "values": data.copy(),
                    "evaluation_mask": analysis_mask.copy(),
                    "record": record,
                }
            elif (
                map_example is not None
                and record.kind == "chimap"
                and record.configuration == configurations[1]
                and record.participant == map_example["participant"]
                and record.method == map_example["method"]
            ):
                map_example["opt_values"] = data.copy()

        for (kind, configuration), method_maps in participant_maps.items():
            method_maps.sort(key=lambda item: item[0])
            for first_index, second_index in pairwise_indices(len(method_maps)):
                first_method, first_data, _ = method_maps[first_index]
                second_method, second_data, _ = method_maps[second_index]
                overlap = (
                    analysis_mask
                    & np.isfinite(first_data)
                    & np.isfinite(second_data)
                )
                agreement = pairwise_agreement(
                    first_data[overlap], second_data[overlap]
                )
                agreement_rows.append(
                    {
                        "participant": participant,
                        "kind": kind,
                        "configuration": configuration,
                        "method_a": first_method,
                        "method_b": second_method,
                        **agreement,
                    }
                )

        for kind in ("chimap", "localfield"):
            for configuration in configurations:
                method_values = [
                    row
                    for row in roi_rows
                    if row["participant"] == participant
                    and row["kind"] == kind
                    and row["configuration"] == configuration
                ]
                for measure in ("gm_mean", "wm_mean", "gm_minus_wm"):
                    samples = np.asarray(
                        [row[measure] for row in method_values], dtype=float
                    )
                    samples = samples[np.isfinite(samples)]
                    dispersion_rows.append(
                        {
                            "participant": participant,
                            "kind": kind,
                            "configuration": configuration,
                            "measure": measure,
                            "n_methods": int(samples.size),
                            "algorithm_sd": (
                                float(np.std(samples, ddof=1))
                                if samples.size > 1
                                else np.nan
                            ),
                            "algorithm_range": (
                                float(np.ptp(samples)) if samples.size else np.nan
                            ),
                        }
                    )

    roi = pd.DataFrame(
        roi_rows,
        columns=[
            "participant",
            "kind",
            "method",
            "configuration",
            "units",
            "gm_n_mask",
            "gm_n_finite",
            "gm_finite_fraction",
            "gm_zero_fraction",
            "gm_mean",
            "gm_sd",
            "wm_n_mask",
            "wm_n_finite",
            "wm_finite_fraction",
            "wm_zero_fraction",
            "wm_mean",
            "wm_sd",
            "gm_wm_overlap_voxels",
            "gm_minus_wm",
        ],
    )
    agreement = pd.DataFrame(
        agreement_rows,
        columns=[
            "participant",
            "kind",
            "configuration",
            "method_a",
            "method_b",
            "n_voxels",
            "pearson_r",
            "mae",
            "rmse",
        ],
    )
    dispersion = pd.DataFrame(
        dispersion_rows,
        columns=[
            "participant",
            "kind",
            "configuration",
            "measure",
            "n_methods",
            "algorithm_sd",
            "algorithm_range",
        ],
    )
    paired_rows: list[dict[str, object]] = []
    if not roi.empty and "method" in roi:
        for (kind, method), subset in roi.groupby(["kind", "method"], sort=True):
            for measure in ("gm_mean", "wm_mean", "gm_minus_wm"):
                paired = subset.pivot(
                    index="participant", columns="configuration", values=measure
                )
                if not all(name in paired.columns for name in configurations):
                    continue
                participants_with_pair = paired.dropna(
                    subset=configurations
                ).index.astype(str).tolist()
                stats_row = paired_comparison(
                    participants_with_pair,
                    paired.loc[participants_with_pair, configurations[0]].tolist(),
                    paired.loc[participants_with_pair, configurations[1]].tolist(),
                )
                paired_rows.append(
                    {
                        "kind": kind,
                        "method": method,
                        "measure": measure,
                        "configuration_a": configurations[0],
                        "configuration_b": configurations[1],
                        **stats_row,
                    }
                )
    if paired_rows:
        paired_df = pd.DataFrame(paired_rows)
        paired_df["paired_t_p_holm"] = holm_adjust(
            paired_df["paired_t_p"].astype(float).tolist()
        )
    else:
        paired_df = pd.DataFrame(
            columns=[
                "kind",
                "method",
                "measure",
                "configuration_a",
                "configuration_b",
                "n",
                "mean_def",
                "mean_opt",
                "mean_difference_opt_minus_def",
                "sd_difference",
                "ci95_low",
                "ci95_high",
                "cohen_dz",
                "paired_t_p",
                "wilcoxon_p",
                "paired_t_p_holm",
            ]
        )

    roi.to_csv(output_dir / "roi_metrics.csv", index=False)
    agreement.to_csv(output_dir / "map_agreement.csv", index=False)
    dispersion.to_csv(output_dir / "algorithm_dispersion.csv", index=False)
    paired_df.to_csv(output_dir / "paired_comparisons.csv", index=False)
    issues_df = pd.DataFrame(analysis_issues, columns=["participant", "issue"])
    issues_df.to_csv(output_dir / "analysis_issues.csv", index=False)

    return {
        "inventory": pd.DataFrame(inventory_rows),
        "image_qc": pd.DataFrame(qc_rows),
        "roi_metrics": roi,
        "map_agreement": agreement,
        "algorithm_dispersion": dispersion,
        "paired_comparisons": paired_df,
        "analysis_issues": issues_df,
        "map_example": map_example,
        "valid_map_count": valid_map_count,
        "expected_map_count": len(_expected_map_records(records)),
        "participants": participants,
        "data_root": Path(data_root),
    }
