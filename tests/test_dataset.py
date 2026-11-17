from pathlib import Path

import nibabel as nib
import numpy as np

from spinal_cord_qsm_benchmark.dataset import (
    ImageRecord,
    ImageGeometryError,
    assert_same_grid,
    discover_dataset,
    image_geometry,
    inspect_images,
)


def _save(path: Path, affine: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(
            np.ones((4, 5, 6), dtype=np.float32),
            np.eye(4) if affine is None else affine,
        ),
        path,
    )


def test_discovery_records_missing_expected_maps(tmp_path: Path) -> None:
    records, inventory = discover_dataset(
        tmp_path,
        ["sub-01"],
        {"chimap": ["tkd"], "localfield": ["pdf"]},
        ["def", "opt"],
    )
    chi = next(
        item
        for item in records
        if item.kind == "chimap"
        and item.method == "tkd"
        and item.configuration == "def"
    )
    assert chi.path == (
        tmp_path / "chimap" / "sub-01" / "def_tkd" / "sub-01_desc-chimap.nii.gz"
    )
    assert len(records) == 9
    assert all(item["status"] == "missing" for item in inventory)


def test_geometry_reports_spacing_orientation_and_affine(tmp_path: Path) -> None:
    path = tmp_path / "image.nii.gz"
    _save(path)
    image = nib.load(path)
    geometry = image_geometry(image)
    assert geometry["shape"] == (4, 5, 6)
    assert geometry["orientation"] == "RAS"
    assert geometry["voxel_sizes"] == (1.0, 1.0, 1.0)
    assert_same_grid(image, image)

    changed = np.eye(4)
    changed[0, 3] = 2
    shifted = nib.Nifti1Image(np.ones((4, 5, 6), dtype=np.float32), changed)
    try:
        assert_same_grid(image, shifted)
    except ImageGeometryError as error:
        assert "Affine" in str(error)
    else:
        raise AssertionError("different affine should be rejected")


def test_inspection_prefers_published_cord_mask_as_reference(tmp_path: Path) -> None:
    mask = tmp_path / "sct" / "sub-01" / "sub-01_desc-SC_mask.nii.gz"
    chi = tmp_path / "chimap" / "sub-01" / "def_tkd" / "sub-01_desc-chimap.nii.gz"
    _save(mask)
    _save(chi)
    records = [
        ImageRecord("sub-01", "chimap", chi, "tkd", "def"),
        ImageRecord("sub-01", "cord_mask", mask),
    ]
    qc = inspect_images(records)
    assert all(row["status"] == "ok" for row in qc)
    assert all(row["reference"] == str(mask) for row in qc)


def test_inspection_records_spatial_metadata_for_unsupported_4d_mask(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sct" / "sub-01" / "sub-01_desc-SC_mask.nii.gz"
    path.parent.mkdir(parents=True)
    nib.save(
        nib.Nifti1Image(np.ones((4, 5, 6, 2), dtype=np.float32), np.eye(4)), path
    )
    qc = inspect_images([ImageRecord("sub-01", "cord_mask", path)])
    assert qc[0]["status"] == "invalid"
    assert qc[0]["shape"] == "4x5x6x2"
    assert qc[0]["voxel_sizes"] == "1,1,1"
    assert qc[0]["orientation"] == "RAS"
    assert qc[0]["affine"]
