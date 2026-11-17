# spinal-cord-qsm-benchmark

A reproducible Python framework for comparing published spinal-cord quantitative
susceptibility mapping (QSM) derivatives across brain-recommended and
spinal-cord-optimized processing configurations.

<img width="2670" height="3045" alt="roi_metrics_chimap" src="https://github.com/user-attachments/assets/323236ef-7261-40e9-9092-e5651cb4697e" />
<img width="3570" height="1485" alt="algorithm_agreement" src="https://github.com/user-attachments/assets/4cacd061-2fcf-4518-a2fd-55ef72e05c5a" />


## Motivation

QSM estimates tissue magnetic susceptibility from field perturbations measured
with gradient-echo MRI. Spinal-cord QSM is attractive for studying cord tissue
microstructure, but the cord is small and elongated, lies next to CSF and
vertebrae, and has strong partial-volume and boundary effects. Brain-optimized
algorithms are typically tuned for larger brain fields of view and different
mask geometries. Their regularization and boundary behavior may therefore
affect spinal-cord results differently. Comparing processing choices is
necessary; it does not, by itself, establish which map is accurate.

This project uses the public **NeuroPoly `sc-qsm` dataset**, OpenNeuro
`ds008321` version `1.0.2`
([dataset DOI](https://doi.org/10.18112/openneuro.ds008321.v1.0.2)). It
contains eight participants and published total-field, local-field, and
susceptibility-map derivatives, with five background-field-removal (BGFR)
methods, five dipole-inversion (DI) methods, two configurations, and SCT
spinal-cord, gray-matter, white-matter, and vertebral-level derivatives where
available. The public dataset is derivative data; this workflow does not
assume access to private raw MRI acquisitions.

## Relationship to the upstream work

The upstream [neuropoly/sc_qsm](https://github.com/neuropoly/sc_qsm) project
provides a private-input raw-data processing notebook and a public-data
figures/metrics notebook. The public notebook calculates a weighted
gray-matter/white-matter contrast-to-noise metric by vertebral level and
reproduces the original paper's figures. This benchmark **does not duplicate
that notebook**. It adds derivative inventory and NIfTI geometry checks,
transparent missing-data tracking, ROI means and standard deviations,
voxelwise inter-algorithm agreement, paired participant-level configuration
differences, and an offline HTML report. It neither reimplements the original
weighted CNR nor reproduces the upstream figures.

## Processing and physical interpretation

The published products represent successive QSM stages:

1. **Total field and phase-derived field:** the measured GRE field contains
   both tissue and sources outside the spinal-cord region.
2. **Background-field removal:** PDF, LBV, SHARP, RESHARP, and V-SHARP estimate
   the local field by removing contributions from outside the region of
   interest. Mask erosion and boundary conditions can change which voxels
   remain.
3. **Dipole inversion:** TKD, closed-form L2, FANSI, MEDI, and iLSQR estimate a
   susceptibility distribution from the local field. The dipole kernel has an
   ill-conditioned zero-cone, so regularization and implementation choices
   trade off noise, artifacts, and spatial detail.

`def` denotes the brain-recommended/default parameter configuration and `opt`
the spinal-cord-optimized configuration, following the dataset naming. This
package analyzes existing field and susceptibility maps; it does **not** run
phase unwrapping, background-field removal, dipole inversion, or raw
phase-to-susceptibility reconstruction.

## Metrics

- **Tissue estimates:** participant/method/configuration GM and WM finite-value
  means and sample standard deviations inside each published tissue mask.
- **GM-WM contrast:** signed GM mean minus WM mean, preserving the source map's
  numerical units. No units are assumed when sidecars do not specify them.
- **Coverage diagnostics:** finite fraction and zero fraction within each ROI.
  Finite zero values are retained in mean/SD summaries because a zero may be a
  real value or an algorithm-specific missing voxel; it is not silently
  discarded.
- **Inter-algorithm agreement:** for each participant and configuration, every
  pair of maps is compared in the union of the published GM/WM masks using Pearson correlation, mean
  absolute error (MAE), and root mean squared error (RMSE) on overlapping
  finite voxels. These are agreement measures, not accuracy measures.
- **Measurement consistency:** per-participant dispersion across algorithms
  and paired `opt - def` differences for GM mean, WM mean, and GM-WM contrast.
  The report gives paired mean differences, t-based 95% confidence intervals,
  Cohen's dz, paired t-test p-values, exact signed-rank p-values, and Holm
  adjustment of t-test p-values across the reported ROI comparisons.
- **Image validation:** expected-file presence, 3D dimensions, voxel spacing,
  orientation codes, and affine consistency. Mismatched images are excluded
  from comparisons rather than silently resampled.

The participant is the unit of statistical comparison (`n` is at most eight),
not the voxel. Confidence intervals and tests are exploratory given the small
sample and the assumptions of paired inference. No ground-truth, diagnostic, or
clinical-validity claims are made.

## Installation and data download

Create the environment with Conda:

```powershell
conda env create -f environment.yml
conda activate spinal-cord-qsm-benchmark
```

Or install into an existing Python 3.11+ environment:

```powershell
python -m pip install -e ".[download,test]"
```

Download the public dataset with the same OpenNeuro Python client used by the
upstream public-data notebook:

```powershell
python -c "from openneuro import download; download(dataset='ds008321', target_dir='data')"
```

The dataset is not included in this repository and is never uploaded by the
analysis. The default configuration expects NIfTI derivatives under
`data/derivatives/` with `chimap/`, `localfield/`, `fieldmap/`, and `sct/`
subfolders. Confirm the download location before running; if needed, set
`data_root` in `config.toml` or pass `--data-root` to the CLI.

## Run the analysis

```powershell
scqsm-benchmark --config config.toml
```

Equivalent direct invocation:

```powershell
python -m spinal_cord_qsm_benchmark.cli --config config.toml
```

The command records missing or invalid files and continues with valid
participant data. Use command-line overrides for a different data or output
location:

```powershell
scqsm-benchmark --config config.toml --data-root "D:\datasets\ds008321\derivatives" --output-dir "D:\qsm-results"
```

Outputs are written to `results/` by default:

| Output | Contents |
|---|---|
| `report.html` | Self-contained report with embedded figures, results, methodology, and limitations |
| `inventory.csv` | Expected file inventory, including absent files |
| `image_qc.csv` | Header geometry and compatibility status for each available NIfTI |
| `roi_metrics.csv` | Participant-level ROI measurements |
| `map_agreement.csv` | Pairwise map agreement within the GM/WM mask union |
| `algorithm_dispersion.csv` | Across-method ROI dispersion by participant/configuration |
| `paired_comparisons.csv` | Paired `opt - def` effects, intervals, and tests |
| `analysis_issues.csv` | Mask, image, and analysis issues encountered |
| `figures/` | 300-dpi ROI, algorithm-agreement, and susceptibility-map comparison plots |

If the data are unavailable, the pipeline still writes the expected inventory,
QC status, and a report that clearly states that no measurements were produced.
It never fills in results or anatomical masks.

### Results from the public-data run

The public v1.0.2 derivatives were downloaded and analyzed locally during
development: all **160/160** expected local-field and susceptibility maps and
all eight GM/WM mask pairs were usable. One cord-mask file (`sub-01`) is
4D (`384 x 384 x 16 x 5`) rather than a 3D mask. It is explicitly flagged in
`image_qc.csv`; the pipeline does not guess what its fifth dimension means and
uses the provided GM/WM mask union for map comparisons. All other available
NIfTI headers passed the spatial-grid checks. The dataset sidecars did not
specify the numeric units, so the reported values are labeled only as source
units.

As a compact descriptive summary, the table below averages the eight-subject
paired `opt - def` mean differences across the five algorithms within each
map family. These are averages of algorithm-specific results, not a pooled
estimate or a test of one algorithm family against another.

| Map family | ROI measure | Mean of algorithm-specific paired mean differences (source units) |
|---|---|---:|
| Susceptibility maps (`chimap`) | GM mean | -0.001151 |
| Susceptibility maps (`chimap`) | WM mean | +0.000235 |
| Susceptibility maps (`chimap`) | GM-WM contrast | -0.001386 |
| Local-field maps (`localfield`) | GM mean | +0.272990 |
| Local-field maps (`localfield`) | WM mean | +0.004944 |
| Local-field maps (`localfield`) | GM-WM contrast | +0.268046 |

Mean pairwise voxel correlations over participants and algorithm pairs were
0.809 (`def`) versus 0.849 (`opt`) for susceptibility maps, and 0.197 versus
0.437 for local-field maps. These averages are descriptive; they do not
establish accuracy or superiority. The complete participant-level estimates,
algorithm-specific paired tests, confidence intervals, effect sizes, map
disagreement metrics, and limitations are generated in `results/report.html`
when the pipeline is run.

## Tests

```powershell
python -m pytest
```

## Limitations

There are eight participants, and the published maps are not ground truth.
Inter-method agreement, susceptibility contrast, and parameter sensitivity
describe these reconstructions rather than their absolute error. Segmentation
uncertainty in the provided masks is not estimated. Zero-valued voxels inside
ROIs are ambiguous and therefore retained in the primary summaries while their
fraction is reported separately. Header agreement does not guarantee
anatomical registration quality. Missing files, invalid masks, differing grids,
and unspecified units are reported rather than inferred or silently repaired.

For the original processing and evaluation context, see the upstream
[sc_qsm repository](https://github.com/neuropoly/sc_qsm) and the dataset's
[OpenNeuro record](https://openneuro.org/datasets/ds008321/versions/1.0.2).
