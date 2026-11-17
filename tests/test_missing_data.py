from pathlib import Path

from spinal_cord_qsm_benchmark.analysis import run_analysis
from spinal_cord_qsm_benchmark.report import generate_report
from spinal_cord_qsm_benchmark.visualization import create_figures


def test_missing_public_data_produces_explicit_empty_report(tmp_path: Path) -> None:
    results = run_analysis(
        data_root=tmp_path / "missing-derivatives",
        output_dir=tmp_path / "results",
        participants=["sub-01"],
        algorithms={"chimap": ["tkd"], "localfield": ["pdf"]},
        configurations=["def", "opt"],
    )
    figures = create_figures(results, tmp_path / "results" / "figures")
    report_path = generate_report(
        results, figures, tmp_path / "results" / "report.html"
    )

    assert results["roi_metrics"].empty
    assert (results["inventory"]["status"] == "missing").all()
    assert figures == []
    report = report_path.read_text(encoding="utf-8")
    assert "No valid susceptibility or local-field maps were available" in report
    assert "does not contain fabricated measurements" in report
