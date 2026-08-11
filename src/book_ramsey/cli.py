"""Command-line interface for Book Ramsey construction utilities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .validation import parameters_from_filename, verify_witness_file


def _witness_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        path.glob("R(B*_B*_*).txt"),
        key=lambda witness_path: parameters_from_filename(witness_path),
    )


def _verify(path: Path) -> int:
    files = _witness_files(path)
    if not files:
        print(f"No witness files found in {path}")
        return 2

    failed = False
    print("file\torder\tmax(color 1)\tmax(color 0)\tresult")
    for witness_path in files:
        try:
            result = verify_witness_file(witness_path)
        except (OSError, ValueError) as error:
            failed = True
            print(f"{witness_path.name}\t-\t-\t-\tERROR: {error}")
            continue

        status = "OK" if result.is_witness else "FAILED"
        failed |= not result.is_witness
        print(
            f"{witness_path.name}\t{result.order}\t{result.color_one_max}"
            f"\t{result.color_zero_max}\t{status}"
        )
    return int(failed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="book-ramsey", description="Verify two-color Book Ramsey constructions."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify", help="verify a matrix file or directory")
    verify_parser.add_argument("path", type=Path, help="witness file or directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        return _verify(args.path)
    raise AssertionError(f"unhandled command: {args.command}")
