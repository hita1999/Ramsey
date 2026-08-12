"""Compare reference and integer-bitset Book Ramsey witness validation."""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from book_ramsey.bitset import verify_book_ramsey_witness_bitset
from book_ramsey.validation import (
    parameters_from_filename,
    read_adjacency_matrix,
    verify_book_ramsey_witness,
)

REPOSITORY_ROOT = Path(__file__).parents[1]
DEFAULT_WITNESS = REPOSITORY_ROOT / "decidedRamseyNumber" / "R(B12_B13_50).txt"


def measure(function: Callable[[], Any], iterations: int, repeats: int) -> float:
    """Return the median seconds per call over repeated timing batches."""

    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(iterations):
            function()
        samples.append((time.perf_counter() - started) / iterations)
    return statistics.median(samples)


def parse_args() -> argparse.Namespace:
    """Parse command-line benchmark options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("witness", nargs="?", type=Path, default=DEFAULT_WITNESS)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    """Run the reference and bitset implementations on the same witness."""

    args = parse_args()
    if args.iterations < 1 or args.repeats < 1:
        raise SystemExit("--iterations and --repeats must be positive")

    matrix = read_adjacency_matrix(args.witness)
    first_book, second_book, _ = parameters_from_filename(args.witness)
    def reference_call():
        return verify_book_ramsey_witness(matrix, first_book, second_book)

    def bitset_call():
        return verify_book_ramsey_witness_bitset(matrix, first_book, second_book)

    if reference_call() != bitset_call():
        raise RuntimeError("implementations returned different results")

    reference_seconds = measure(reference_call, args.iterations, args.repeats)
    bitset_seconds = measure(bitset_call, args.iterations, args.repeats)
    print(f"witness: {args.witness}")
    print(f"reference: {reference_seconds * 1_000:.3f} ms/call")
    print(f"bitset:    {bitset_seconds * 1_000:.3f} ms/call")
    print(f"speedup:   {reference_seconds / bitset_seconds:.2f}x")


if __name__ == "__main__":
    main()
