"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tomllib

from .analysis import run_analysis
from .report import generate_report
from .visualization import create_figures


def _read_config(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as stream:
            config = tomllib.load(stream)
    except OSError as error:
        raise ValueError(f"Unable to read config file {path}: {error}") from error
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"Invalid TOML in {path}: {error}") from error

    configurations = config.get("configurations")
    if not isinstance(configurations, list) or set(configurations) != {"def", "opt"}:
        raise ValueError("configurations must contain exactly 'def' and 'opt'")
    participants = config.get("participants")
    algorithms = config.get("algorithms")
    if not isinstance(participants, list) or not participants:
        raise ValueError("participants must be a non-empty list")
    if not isinstance(algorithms, dict) or not all(
        isinstance(algorithms.get(kind), list) and algorithms[kind]
        for kind in ("chimap", "localfield")
    ):
        raise ValueError("algorithms must define non-empty chimap and localfield lists")
    if len(set(participants)) != len(participants):
        raise ValueError("participants must not contain duplicates")
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze published NeuroPoly spinal-cord QSM derivatives."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="TOML configuration file (default: config.toml)",
    )
    parser.add_argument("--data-root", type=Path, help="Override the derivatives directory")
    parser.add_argument("--output-dir", type=Path, help="Override the results directory")
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    try:
        config = _read_config(config_path)
        configured_data_root = Path(str(config["data_root"]))
        configured_output_dir = Path(str(config["output_dir"]))
        data_root = args.data_root or (
            configured_data_root
            if configured_data_root.is_absolute()
            else config_path.parent / configured_data_root
        )
        output_dir = args.output_dir or (
            configured_output_dir
            if configured_output_dir.is_absolute()
            else config_path.parent / configured_output_dir
        )
        results = run_analysis(
            data_root=data_root,
            output_dir=output_dir,
            participants=[str(item) for item in config["participants"]],
            algorithms=config["algorithms"],
            configurations=[str(item) for item in config["configurations"]],
        )
        figures = create_figures(results, output_dir / "figures")
        report_path = generate_report(results, figures, output_dir / "report.html")
    except (OSError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    missing = int((results["inventory"]["status"] == "missing").sum())
    print(
        f"Analyzed {results['valid_map_count']} / {results['expected_map_count']} "
        f"published maps; {missing} expected files are missing."
    )
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
