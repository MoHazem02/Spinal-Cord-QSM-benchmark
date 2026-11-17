"""Dataset discovery and NIfTI geometry checks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np


MAP_SUFFIXES = {"chimap": "chimap", "localfield": "localfield"}
MASK_SUFFIXES = {
    "cord_mask": "SC_mask",
    "gm_mask": "GM_mask",
    "wm_mask": "WM_mask",
    "vertebral_levels": "SC_vertlevels_dseg",
}
FIELD_SUFFIX = "b0_fieldmap"


@dataclass(frozen=True)
class ImageRecord:
    participant: str
    kind: str
    path: Path
    method: str | None = None
    configuration: str | None = None


class ImageGeometryError(ValueError):
    """Raised when NIfTI images do not occupy the same physical grid."""


def discover_dataset(
    data_root: Path,
    participants: Iterable[str],
    algorithms: dict[str, list[str]],
    configurations: Iterable[str],
) -> tuple[list[ImageRecord], list[dict[str, str]]]:
    """Enumerate expected maps and anatomical derivatives, recording absent files."""
    data_root = Path(data_root)
    records: list[ImageRecord] = []
    inventory: list[dict[str, str]] = []

    def add(record: ImageRecord) -> None:
        records.append(record)
        inventory.append(
            {
                "participant": record.participant,
                "kind": record.kind,
                "method": record.method or "",
                "configuration": record.configuration or "",
                "path": str(record.path),
                "status": "present" if record.path.is_file() else "missing",
            }
        )

    for participant in participants:
        for kind, suffix in MAP_SUFFIXES.items():
            for configuration in configurations:
                for method in algorithms.get(kind, []):
                    folder = f"{configuration}_{method}"
                    add(
                        ImageRecord(
                            participant=participant,
                            kind=kind,
                            method=method,
                            configuration=configuration,
                            path=(
                                data_root
                                / kind
                                / participant
                                / folder
                                / f"{participant}_desc-{suffix}.nii.gz"
                            ),
                        )
                    )

        add(
            ImageRecord(
                participant=participant,
                kind="fieldmap",
                path=(
                    data_root
                    / "fieldmap"
                    / participant
                    / f"{participant}_desc-{FIELD_SUFFIX}.nii.gz"
                ),
            )
        )
        for kind, suffix in MASK_SUFFIXES.items():
            add(
                ImageRecord(
                    participant=participant,
                    kind=kind,
                    path=(
                        data_root
                        / "sct"
                        / participant
                        / f"{participant}_desc-{suffix}.nii.gz"
                    ),
                )
            )

    return records, inventory


def image_geometry(image: nib.spatialimages.SpatialImage) -> dict[str, object]:
    """Return spatial metadata, rejecting non-3D or invalid geometry."""
    if len(image.shape) != 3:
        raise ImageGeometryError(f"Expected a 3D image; found shape {image.shape}")

    affine = np.asarray(image.affine, dtype=float)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ImageGeometryError("Image affine is missing or contains non-finite values")
    if abs(float(np.linalg.det(affine[:3, :3]))) < 1e-12:
        raise ImageGeometryError("Image affine is singular")

    voxel_sizes = tuple(float(value) for value in image.header.get_zooms()[:3])
    if len(voxel_sizes) != 3 or not np.isfinite(voxel_sizes).all():
        raise ImageGeometryError("Image voxel spacing is invalid")
    if any(value <= 0 for value in voxel_sizes):
        raise ImageGeometryError(f"Image voxel spacing must be positive: {voxel_sizes}")

    return {
        "shape": tuple(int(value) for value in image.shape),
        "voxel_sizes": voxel_sizes,
        "spatial_unit": image.header.get_xyzt_units()[0] or "unknown",
        "orientation": "".join(nib.aff2axcodes(affine)),
        "affine": affine,
    }


def assert_same_grid(
    reference: nib.spatialimages.SpatialImage,
    candidate: nib.spatialimages.SpatialImage,
    *,
    affine_atol: float = 1e-4,
) -> None:
    """Raise if dimensions, spacing, orientation, or affine differ."""
    ref = image_geometry(reference)
    other = image_geometry(candidate)
    if ref["shape"] != other["shape"]:
        raise ImageGeometryError(f"Shape mismatch: {ref['shape']} != {other['shape']}")
    if ref["spatial_unit"] != other["spatial_unit"]:
        raise ImageGeometryError(
            f"Spatial-unit mismatch: {ref['spatial_unit']} != {other['spatial_unit']}"
        )
    if not np.allclose(
        ref["voxel_sizes"], other["voxel_sizes"], rtol=0, atol=1e-5
    ):
        raise ImageGeometryError(
            f"Voxel-spacing mismatch: {ref['voxel_sizes']} != {other['voxel_sizes']}"
        )
    if ref["orientation"] != other["orientation"]:
        raise ImageGeometryError(
            f"Orientation mismatch: {ref['orientation']} != {other['orientation']}"
        )
    if not np.allclose(ref["affine"], other["affine"], rtol=0, atol=affine_atol):
        raise ImageGeometryError("Affine matrices differ")


def inspect_images(
    records: list[ImageRecord],
) -> list[dict[str, object]]:
    """Load headers and validate every present image against a participant reference."""
    by_participant: dict[str, list[ImageRecord]] = {}
    for record in records:
        by_participant.setdefault(record.participant, []).append(record)

    qc_rows: list[dict[str, object]] = []
    for participant, participant_records in by_participant.items():
        reference: nib.spatialimages.SpatialImage | None = None
        reference_path: Path | None = None
        ordered_records = sorted(
            participant_records,
            key=lambda item: (
                item.kind != "cord_mask",
                item.kind not in ("chimap", "localfield"),
                str(item.path),
            ),
        )

        for record in ordered_records:
            row: dict[str, object] = {
                "participant": participant,
                "kind": record.kind,
                "method": record.method or "",
                "configuration": record.configuration or "",
                "path": str(record.path),
                "reference": "",
                "status": "missing",
                "shape": "",
                "voxel_sizes": "",
                "spatial_unit": "",
                "orientation": "",
                "affine": "",
                "issue": "",
            }
            if not record.path.is_file():
                qc_rows.append(row)
                continue
            try:
                image = nib.load(str(record.path))
            except (OSError, ValueError, EOFError, nib.filebasedimages.ImageFileError) as error:
                row["status"] = "invalid"
                row["issue"] = str(error)
                qc_rows.append(row)
                continue

            row["shape"] = "x".join(str(value) for value in image.shape)
            row["voxel_sizes"] = ",".join(
                f"{value:.6g}" for value in image.header.get_zooms()[:3]
            )
            row["spatial_unit"] = image.header.get_xyzt_units()[0] or "unknown"
            affine = np.asarray(image.affine, dtype=float)
            row["affine"] = (
                json.dumps(affine.round(6).tolist(), separators=(",", ":"))
                if np.isfinite(affine).all()
                else "non-finite"
            )
            try:
                geometry = image_geometry(image)
            except ImageGeometryError as error:
                row["status"] = "invalid"
                row["issue"] = str(error)
                try:
                    row["orientation"] = "".join(nib.aff2axcodes(image.affine))
                except (TypeError, ValueError):
                    row["orientation"] = "unknown"
                qc_rows.append(row)
                continue

            row["orientation"] = geometry["orientation"]

            if reference is None:
                reference = image
                reference_path = record.path
            row["reference"] = str(reference_path)
            try:
                assert_same_grid(reference, image)
            except ImageGeometryError as error:
                row["status"] = "mismatch"
                row["issue"] = str(error)
            else:
                row["status"] = "ok"
            qc_rows.append(row)

    return qc_rows
