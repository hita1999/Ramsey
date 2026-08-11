"""Independent validation of adjacency matrices used as Book Ramsey witnesses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

AdjacencyMatrix = tuple[tuple[int, ...], ...]

_WITNESS_FILENAME = re.compile(r"R\(B(?P<first>\d+)_B(?P<second>\d+)_(?P<order>\d+)\)\.txt$")


class MatrixFormatError(ValueError):
    """Raised when a text file is not a simple undirected 0/1 adjacency matrix."""


@dataclass(frozen=True)
class VerificationResult:
    """Summary of a verified candidate for ``R(B_first, B_second)``."""

    order: int
    first_book: int
    second_book: int
    color_one_max: int
    color_zero_max: int
    color_one_spine: tuple[int, int] | None
    color_zero_spine: tuple[int, int] | None

    @property
    def is_witness(self) -> bool:
        return (
            self.color_one_max < self.first_book
            and self.color_zero_max < self.second_book
        )

    @property
    def ramsey_lower_bound(self) -> int:
        return self.order + 1


def read_adjacency_matrix(path: str | Path) -> AdjacencyMatrix:
    """Read a matrix whose non-empty lines consist only of the characters 0 and 1."""

    source = Path(path)
    rows: list[tuple[int, ...]] = []
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        if any(character not in "01" for character in line):
            raise MatrixFormatError(f"{source}:{line_number}: expected only 0 and 1")
        rows.append(tuple(int(character) for character in line))

    matrix = tuple(rows)
    validate_adjacency_matrix(matrix, source=source)
    return matrix


def validate_adjacency_matrix(
    matrix: Sequence[Sequence[int]], *, source: str | Path = "matrix"
) -> None:
    """Validate that ``matrix`` represents a simple undirected graph."""

    label = str(source)
    order = len(matrix)
    if order == 0:
        raise MatrixFormatError(f"{label}: matrix is empty")
    if any(len(row) != order for row in matrix):
        lengths = sorted({len(row) for row in matrix})
        raise MatrixFormatError(
            f"{label}: expected a square {order}x{order} matrix; row lengths are {lengths}"
        )

    for vertex in range(order):
        if matrix[vertex][vertex] != 0:
            raise MatrixFormatError(f"{label}: diagonal entry ({vertex}, {vertex}) must be 0")
        for other in range(vertex):
            if matrix[vertex][other] not in (0, 1):
                raise MatrixFormatError(
                    f"{label}: entry ({vertex}, {other}) is not a binary value"
                )
            if matrix[vertex][other] != matrix[other][vertex]:
                raise MatrixFormatError(
                    f"{label}: entries ({vertex}, {other}) and ({other}, {vertex}) differ"
                )


def maximum_book_size(
    matrix: Sequence[Sequence[int]], color: int
) -> tuple[int, tuple[int, int] | None]:
    """Return the largest monochromatic book and one of its spine edges."""

    if color not in (0, 1):
        raise ValueError("color must be 0 or 1")

    order = len(matrix)
    maximum = 0
    maximum_spine: tuple[int, int] | None = None
    for first in range(order - 1):
        for second in range(first + 1, order):
            if matrix[first][second] != color:
                continue
            pages = sum(
                matrix[first][vertex] == color and matrix[second][vertex] == color
                for vertex in range(order)
                if vertex not in (first, second)
            )
            if pages > maximum or maximum_spine is None:
                maximum = pages
                maximum_spine = (first, second)
    return maximum, maximum_spine


def verify_book_ramsey_witness(
    matrix: Sequence[Sequence[int]], first_book: int, second_book: int
) -> VerificationResult:
    """Check whether ``matrix`` avoids ``B_first`` in color 1 and ``B_second`` in color 0."""

    if first_book < 1 or second_book < 1:
        raise ValueError("book sizes must be positive")
    validate_adjacency_matrix(matrix)
    color_one_max, color_one_spine = maximum_book_size(matrix, color=1)
    color_zero_max, color_zero_spine = maximum_book_size(matrix, color=0)
    return VerificationResult(
        order=len(matrix),
        first_book=first_book,
        second_book=second_book,
        color_one_max=color_one_max,
        color_zero_max=color_zero_max,
        color_one_spine=color_one_spine,
        color_zero_spine=color_zero_spine,
    )


def parameters_from_filename(path: str | Path) -> tuple[int, int, int]:
    """Extract ``(first_book, second_book, order)`` from a decided-matrix filename."""

    source = Path(path)
    match = _WITNESS_FILENAME.fullmatch(source.name)
    if match is None:
        raise ValueError(f"unsupported witness filename: {source.name}")
    return tuple(int(match.group(name)) for name in ("first", "second", "order"))


def verify_witness_file(path: str | Path) -> VerificationResult:
    """Read and verify a witness file whose parameters are encoded in its filename."""

    first_book, second_book, expected_order = parameters_from_filename(path)
    matrix = read_adjacency_matrix(path)
    if len(matrix) != expected_order:
        raise MatrixFormatError(
            f"{path}: filename declares order {expected_order}, but matrix has order {len(matrix)}"
        )
    return verify_book_ramsey_witness(matrix, first_book, second_book)
