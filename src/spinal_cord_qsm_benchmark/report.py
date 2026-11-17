"""Generate a self-contained HTML summary without external assets."""

from __future__ import annotations

import base64
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    if columns is not None:
        available = [column for column in columns if column in frame.columns]
        frame = frame.loc[:, available]
    if frame.empty:
        return "<p>No records were available for this analysis.</p>"
    return frame.to_html(index=False, border=0, classes="data-table", na_rep="NA", float_format=lambda value: f"{value:.5g}")


def _image(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<figure><img src="data:image/png;base64,{encoded}" alt="{escape(path.stem)}">'
        f"<figcaption>{escape(path.stem.replace('_', ' ').title())}</figcaption></figure>"
    )


def generate_report(
    results: dict[str, Any], figure_paths: list[Path], output_path: Path
) -> Path:
    """Write an offline-readable report containing figures and all primary tables."""
    inventory = results["inventory"]
    qc = results["image_qc"]
    missing_count = int((inventory["status"] == "missing").sum()) if not inventory.empty else 0
    qc_counts = (
        qc["status"].value_counts().to_dict()
        if not qc.empty
        else {}
    )
    analyzed_participants = (
        results["roi_metrics"]["participant"].nunique()
        if not results["roi_metrics"].empty
        else 0
    )
    issue_text = (
        results["analysis_issues"].to_html(
            index=False, border=0, classes="data-table", na_rep="NA"
        )
        if not results["analysis_issues"].empty
        else "<p>No image-level analysis issues were recorded.</p>"
    )
    figures = "\n".join(_image(path) for path in figure_paths)
    figure_section = figures or "<p>No figures were generated because no valid maps were available.</p>"

    conclusion = (
        "Published maps were available for analysis. Results describe differences and "
        "agreement among these reconstructions in this dataset; they do not establish "
        "accuracy, ground truth, or clinical validity."
        if results["valid_map_count"]
        else "No valid susceptibility or local-field maps were available to analyze. "
        "The report contains the expected-file inventory and validation results only; "
        "it does not contain fabricated measurements."
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spinal-cord QSM benchmark report</title>
<style>
body {{ font: 16px/1.55 system-ui, sans-serif; max-width: 1180px; margin: 2rem auto; padding: 0 1rem; color: #17212b; }}
h1, h2 {{ color: #173a5e; }} h2 {{ border-bottom: 1px solid #d5dfe8; padding-bottom: .35rem; margin-top: 2.2rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: .88rem; display: block; overflow-x: auto; margin: 1rem 0 2rem; }}
th, td {{ border: 1px solid #dbe2e8; padding: .4rem .55rem; text-align: left; white-space: nowrap; }}
th {{ background: #edf3f7; }} figure {{ margin: 1.5rem 0; }} img {{ max-width: 100%; height: auto; border: 1px solid #dbe2e8; }}
figcaption {{ color: #506070; font-size: .9rem; }} .note {{ background: #f0f5f8; padding: 1rem; border-left: 4px solid #336699; }}
</style>
</head>
<body>
<h1>Spinal-cord QSM benchmark</h1>
<p>Reproducible, participant-level evaluation of published susceptibility maps and
local-field maps from NeuroPoly/OpenNeuro ds008321 v1.0.2.</p>
<p class="note"><strong>Dataset access:</strong> <a href="https://doi.org/10.18112/openneuro.ds008321.v1.0.2">OpenNeuro ds008321 v1.0.2</a>.
This analysis uses published derivatives only; it does not reconstruct QSM from raw phase.</p>

<h2>Dataset description</h2>
<p>The public dataset contains eight participants. Published derivatives include
total field maps, local-field maps, susceptibility maps, SCT spinal-cord/gray-matter/
white-matter masks, and vertebral-level labels. Five background-field-removal methods
(PDF, LBV, SHARP, RESHARP, V-SHARP) and five dipole-inversion methods (TKD,
closed-form L2, FANSI, MEDI, iLSQR) are distributed with brain-recommended (<code>def</code>)
and spinal-cord-optimized (<code>opt</code>) configurations where available.</p>
<p>The upstream <a href="https://github.com/neuropoly/sc_qsm">sc_qsm</a> repository
already reproduces weighted GM-WM contrast-to-noise metrics by vertebral level.
This project complements those analyses with derivative inventory, spatial-header QC,
voxelwise inter-algorithm agreement, configuration differences, and paired ROI estimates.</p>

<h2>Scientific background</h2>
<p>QSM estimates tissue magnetic susceptibility from GRE-induced field perturbations.
Background-field removal separates local tissue fields from contributions external to
the region of interest; dipole inversion estimates susceptibility from that local field.
Dipole inversion is ill-conditioned around the dipole kernel's zero-cone, so algorithm
regularization and parameter choices can alter the reconstructed map. Spinal-cord
geometry is small, elongated, and adjacent to CSF and vertebral structures; brain-tuned
boundary assumptions and regularization therefore need not transfer directly.</p>

<h2>Methodology</h2>
<ul>
<li>Discover expected participant, method, configuration, map, fieldmap, and mask paths.</li>
<li>Validate 3D dimensions, voxel spacing, orientation codes, and affine matrices against
a compatible participant reference grid. Incompatible images are excluded, not silently resampled.</li>
<li>Use the published GM and WM masks for ROI estimates, and their union as the map-comparison
region. Compute finite-value
mean and sample SD, signed GM-WM contrast, finite coverage, and zero fraction per ROI.
Finite zero values are retained; zero fraction is reported separately because zeros may
represent either valid values or algorithm-specific missing voxels.</li>
<li>Within each participant and configuration, compare algorithms pairwise inside the
union of the published GM and WM masks using Pearson correlation, MAE, and RMSE on
overlapping finite voxels.</li>
<li>Compare <code>opt</code> against <code>def</code> within participants for each
algorithm. Report paired mean differences, t-based 95% confidence intervals, Cohen's
dz, paired t-test p-values, exact Wilcoxon p-values where defined, and Holm-adjusted
t-test p-values across the reported ROI comparisons.</li>
</ul>
<p>No resampling, raw phase-to-susceptibility reconstruction, fabricated masks, or
ground-truth comparisons are performed.</p>

<h2>Results</h2>
<p>Analyzed valid map files: {results["valid_map_count"]} / {results["expected_map_count"]}.
Participants with ROI measurements: {analyzed_participants} / {len(results["participants"])}.
Expected files missing: {missing_count}. Image QC statuses: {escape(str(qc_counts))}.</p>
<p>{escape(conclusion)}</p>
<h3>Participant-level ROI measurements</h3>
{_table(results["roi_metrics"], ["participant", "kind", "method", "configuration", "units", "gm_mean", "gm_sd", "gm_finite_fraction", "gm_zero_fraction", "wm_mean", "wm_sd", "wm_finite_fraction", "wm_zero_fraction", "gm_minus_wm"])}
<h3>Paired configuration comparisons (opt - def)</h3>
{_table(results["paired_comparisons"], ["kind", "method", "measure", "n", "mean_difference_opt_minus_def", "ci95_low", "ci95_high", "cohen_dz", "paired_t_p", "paired_t_p_holm", "wilcoxon_p"])}
<h3>Inter-algorithm map agreement</h3>
{_table(results["map_agreement"], ["participant", "kind", "configuration", "method_a", "method_b", "n_voxels", "pearson_r", "mae", "rmse"])}
<h3>Inter-algorithm ROI dispersion</h3>
{_table(results["algorithm_dispersion"], ["participant", "kind", "configuration", "measure", "n_methods", "algorithm_sd", "algorithm_range"])}

<h2>Figures</h2>
{figure_section}

<h2>Data validation and missingness</h2>
<p>Inventory and geometry details are also available in the CSV files saved alongside
this report. Images with different dimensions, spacing, orientation, or affine are
flagged and excluded from quantitative comparisons. Missing maps, masks, or invalid
headers remain explicit in the inventory/QC outputs.</p>
{issue_text}

<h2>Limitations and conclusions</h2>
<ul>
<li>Eight participants provide limited precision and power. Confidence intervals and
p-values are exploratory; participant-level estimates are shown and multiplicity is
addressed only across the reported paired t-tests.</li>
<li>These maps are reconstructions, not ground truth. Agreement and GM-WM contrast
are not measures of absolute accuracy, and no clinical-validity conclusion follows.</li>
<li>Published tissue masks may carry segmentation uncertainty; no new masks are inferred.
The cord mask is checked but is not collapsed or interpreted when it is not a compatible
3D image; the GM/WM mask union defines the comparison region.</li>
<li>Zeros within published ROI masks are retained in mean/SD summaries because their
meaning is not identifiable from image values alone. Their fraction is reported as a
coverage diagnostic, not treated as a definitive invalid-voxel label.</li>
<li>Header checks do not prove anatomical registration quality. Differing grids are
excluded rather than transformed; reported values preserve the source map's units,
which are shown only when supplied in sidecar metadata.</li>
</ul>
<p><strong>Conclusion:</strong> {escape(conclusion)} Interpret every comparison as
descriptive evidence about these published processing outputs, not proof that one
configuration is accurate or clinically preferable.</p>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")
    return output_path
