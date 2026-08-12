"""Pure-Python bitset evaluation for two-color Book Ramsey graphs.

Each adjacency row is stored as a Python :class:`int`.  Intersections of two
neighborhoods then become one bitwise AND and the number of common neighbors
is obtained with :meth:`int.bit_count`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .validation import VerificationResult, validate_adjacency_matrix


@dataclass(frozen=True, slots=True)
class BookViolation:
    """A spine edge whose monochromatic common neighborhood is too large."""

    color: int
    required_pages: int
    actual_pages: int
    spine: tuple[int, int]


@dataclass(frozen=True, slots=True)
class BitsetGraph:
    """A validated two-color complete graph represented by integer row bitsets."""

    order: int
    color_one_rows: tuple[int, ...]
    color_zero_rows: tuple[int, ...]

    @classmethod
    def from_adjacency_matrix(cls, matrix: Sequence[Sequence[int]]) -> BitsetGraph:
        """Validate and convert a simple undirected 0/1 adjacency matrix."""

        validate_adjacency_matrix(matrix)
        order = len(matrix)
        universe = (1 << order) - 1
        color_one_rows: list[int] = []
        color_zero_rows: list[int] = []

        for vertex, matrix_row in enumerate(matrix):
            row = 0
            for neighbor, value in enumerate(matrix_row):
                if value == 1:
                    row |= 1 << neighbor
            color_one_rows.append(row)
            color_zero_rows.append(universe & ~(row | (1 << vertex)))

        return cls(
            order=order,
            color_one_rows=tuple(color_one_rows),
            color_zero_rows=tuple(color_zero_rows),
        )

    def rows_for_color(self, color: int) -> tuple[int, ...]:
        """Return the neighborhood bitsets for ``color`` (zero or one)."""

        if color == 1:
            return self.color_one_rows
        if color == 0:
            return self.color_zero_rows
        raise ValueError("color must be 0 or 1")

    def maximum_book_size(self, color: int) -> tuple[int, tuple[int, int] | None]:
        """Return the largest monochromatic book and one of its spine edges."""

        rows = self.rows_for_color(color)
        maximum = 0
        maximum_spine: tuple[int, int] | None = None

        for first in range(self.order - 1):
            first_row = rows[first]
            for second in range(first + 1, self.order):
                if not first_row & (1 << second):
                    continue
                pages = (first_row & rows[second]).bit_count()
                if pages > maximum or maximum_spine is None:
                    maximum = pages
                    maximum_spine = (first, second)

        return maximum, maximum_spine

    def find_book_violation(self, color: int, book_size: int) -> BookViolation | None:
        """Return the first monochromatic ``B_book_size`` found, if one exists.

        The search stops at the first spine with at least ``book_size`` common
        neighbors.  Use :meth:`maximum_book_size` when the exact maximum is
        required.
        """

        if book_size < 1:
            raise ValueError("book size must be positive")

        rows = self.rows_for_color(color)
        for first in range(self.order - 1):
            first_row = rows[first]
            for second in range(first + 1, self.order):
                if not first_row & (1 << second):
                    continue
                pages = (first_row & rows[second]).bit_count()
                if pages >= book_size:
                    return BookViolation(
                        color=color,
                        required_pages=book_size,
                        actual_pages=pages,
                        spine=(first, second),
                    )
        return None


def adjacency_rows_to_bitsets(
    matrix: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return ``(color_one_rows, color_zero_rows)`` for a validated matrix."""

    graph = BitsetGraph.from_adjacency_matrix(matrix)
    return graph.color_one_rows, graph.color_zero_rows


def maximum_book_size_bitset(
    graph: BitsetGraph, color: int
) -> tuple[int, tuple[int, int] | None]:
    """Return the largest monochromatic book using integer bit operations."""

    return graph.maximum_book_size(color)


def find_book_violation_bitset(
    graph: BitsetGraph, color: int, book_size: int
) -> BookViolation | None:
    """Return the first forbidden book found in ``graph``, or ``None``."""

    return graph.find_book_violation(color, book_size)


def verify_book_ramsey_witness_bitset(
    matrix: Sequence[Sequence[int]], first_book: int, second_book: int
) -> VerificationResult:
    """Verify a Book Ramsey witness using Python integer neighborhood bitsets."""

    if first_book < 1 or second_book < 1:
        raise ValueError("book sizes must be positive")

    graph = BitsetGraph.from_adjacency_matrix(matrix)
    color_one_max, color_one_spine = graph.maximum_book_size(color=1)
    color_zero_max, color_zero_spine = graph.maximum_book_size(color=0)
    return VerificationResult(
        order=graph.order,
        first_book=first_book,
        second_book=second_book,
        color_one_max=color_one_max,
        color_zero_max=color_zero_max,
        color_one_spine=color_one_spine,
        color_zero_spine=color_zero_spine,
    )
