"""Command-line interface for Book Ramsey construction utilities."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from pathlib import Path

from .block_constructions import TwoBlockCyclicFactory
from .search_engine import (
    EdgeBitsFactory,
    MatrixCandidateEvaluator,
    SearchConfig,
    run_search,
    write_search_result,
)
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


def _load_factory(specification: str, order: int):
    if specification == "edge-bits":
        return EdgeBitsFactory(order)
    if specification.startswith("two-block:"):
        parts = specification.split(":")
        if len(parts) not in (3, 4):
            raise ValueError("two-block factory must be two-block:FIRST:SECOND[:SHIFT]")
        try:
            first_size, second_size = (int(value) for value in parts[1:3])
            cross_shift = int(parts[3]) if len(parts) == 4 else 1
        except ValueError as error:
            raise ValueError("two-block sizes and shift must be integers") from error
        factory = TwoBlockCyclicFactory(first_size, second_size, cross_shift)
        if factory.order != order:
            raise ValueError(
                f"--order {order} does not match two-block order {factory.order}"
            )
        return factory
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("factory must be 'edge-bits' or an import path like module:factory")
    factory = getattr(importlib.import_module(module_name), attribute_name)
    if not callable(factory):
        raise ValueError(f"factory is not callable: {specification}")
    return factory


def _write_matrix(path: Path, matrix: Sequence[Sequence[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join("".join(str(value) for value in row) + "\n" for row in matrix),
        encoding="utf-8",
    )


def _search(args: argparse.Namespace) -> int:
    try:
        factory = _load_factory(args.factory, args.order)
        if hasattr(factory, "space_size") and args.stop > factory.space_size:
            raise ValueError(
                f"--stop {args.stop} exceeds factory space size {factory.space_size}"
            )
        config = SearchConfig(
            first_book=args.first_book,
            second_book=args.second_book,
            order=args.order,
            start=args.start,
            stop=args.stop,
            workers=args.workers,
            chunk_size=args.chunk_size,
            checkpoint_path=args.checkpoint,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
            label=args.label or args.factory,
            repository=Path.cwd(),
        )
        outcome = run_search(
            MatrixCandidateEvaluator(
                factory,
                args.first_book,
                args.second_book,
                expected_order=args.order,
            ),
            config,
        )
    except (ImportError, AttributeError, OSError, ValueError) as error:
        print(f"search error: {error}", file=sys.stderr)
        return 2

    if args.result_json is not None:
        write_search_result(args.result_json, outcome)

    metadata = outcome.metadata
    print(
        f"range=[{metadata.effective_start}, {metadata.next_index}) "
        f"examined={metadata.candidates_examined} "
        f"elapsed={metadata.elapsed_seconds:.6f}s "
        f"rate={metadata.candidates_examined / metadata.elapsed_seconds:.2f}/s"
        if metadata.elapsed_seconds
        else f"range=[{metadata.effective_start}, {metadata.next_index}) examined=0"
    )
    if outcome.best is not None:
        score = outcome.best.score
        print(
            f"best={outcome.best.index} violations={score.violating_spines} "
            f"excess_pages={score.excess_pages} "
            f"max_books=({score.color_one.maximum_pages},"
            f"{score.color_zero.maximum_pages})"
        )
    if outcome.witness is None:
        print("not found")
        return 1

    print(f"found candidate {outcome.witness.index}")
    if args.witness_output is not None:
        _write_matrix(args.witness_output, factory(outcome.witness.index))
        print(f"witness written to {args.witness_output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="book-ramsey", description="Verify two-color Book Ramsey constructions."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify", help="verify a matrix file or directory")
    verify_parser.add_argument("path", type=Path, help="witness file or directory")
    search_parser = subparsers.add_parser(
        "search", help="search an integer-indexed family of candidate matrices"
    )
    search_parser.add_argument("--first-book", type=int, required=True)
    search_parser.add_argument("--second-book", type=int, required=True)
    search_parser.add_argument("--order", type=int, required=True)
    search_parser.add_argument("--start", type=int, default=0)
    search_parser.add_argument(
        "--stop", type=int, required=True, help="exclusive end of the candidate range"
    )
    search_parser.add_argument("--workers", type=int, default=1)
    search_parser.add_argument("--chunk-size", type=int, default=256)
    search_parser.add_argument("--checkpoint", type=Path)
    search_parser.add_argument("--checkpoint-every", type=int, default=10_000)
    search_parser.add_argument("--resume", action="store_true")
    search_parser.add_argument(
        "--factory",
        default="edge-bits",
        help=(
            "'edge-bits', 'two-block:FIRST:SECOND[:SHIFT]', or a one-argument "
            "callable as module:attribute"
        ),
    )
    search_parser.add_argument("--label")
    search_parser.add_argument("--result-json", type=Path)
    search_parser.add_argument("--witness-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        return _verify(args.path)
    if args.command == "search":
        return _search(args)
    raise AssertionError(f"unhandled command: {args.command}")
