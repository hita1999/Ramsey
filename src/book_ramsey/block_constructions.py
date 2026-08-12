"""Block-cyclic constructions, including blocks of unequal orders."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from .validation import AdjacencyMatrix, validate_adjacency_matrix

BinaryPattern = tuple[int, ...]
BlockPosition = tuple[int, int]


@dataclass(frozen=True)
class TwoBlockParameters:
    """Decoded binary parameters for a two-block cyclic construction."""

    first_diagonal: int
    cross_block: int
    second_diagonal: int


@dataclass(frozen=True, slots=True)
class TwoBlockCyclicFactory:
    """Pickleable integer-indexed factory for an unequal two-block family."""

    first_size: int
    second_size: int
    cross_shift: int = 1

    def __post_init__(self) -> None:
        if self.first_size < 1 or self.second_size < 1:
            raise ValueError("block sizes must be positive")

    @property
    def order(self) -> int:
        return self.first_size + self.second_size

    @property
    def space_size(self) -> int:
        return 1 << two_block_search_dimension(self.first_size, self.second_size)

    def __call__(self, candidate_index: int) -> AdjacencyMatrix:
        return build_two_block_candidate(
            self.first_size,
            self.second_size,
            candidate_index,
            cross_shift=self.cross_shift,
        )


def _validate_binary_pattern(pattern: Sequence[int], expected_length: int, name: str) -> None:
    if len(pattern) != expected_length:
        raise ValueError(f"{name} must contain {expected_length} entries")
    if any(value not in (0, 1) for value in pattern):
        raise ValueError(f"{name} must contain only 0 and 1")


def integer_to_pattern(length: int, value: int) -> BinaryPattern:
    """Return ``length`` bits in most-significant-bit-first order."""

    if length < 0:
        raise ValueError("length must be non-negative")
    if value < 0 or value >= 1 << length:
        raise ValueError(f"value must be in range(2**{length})")
    return tuple((value >> shift) & 1 for shift in range(length - 1, -1, -1))


def symmetric_circulant_column(order: int, value: int) -> BinaryPattern:
    """Construct the first column of a simple symmetric circulant graph.

    The independent bits correspond to cyclic distances ``1`` through
    ``floor(order / 2)``. The diagonal entry at distance zero is always zero.
    """

    if order < 1:
        raise ValueError("order must be positive")
    independent = integer_to_pattern(order // 2, value)
    column = [0] * order
    for distance, bit in enumerate(independent, 1):
        column[distance] = bit
        column[-distance] = bit
    return tuple(column)


def cyclic_block(
    row_count: int,
    column_count: int,
    pattern: Sequence[int],
    *,
    shift: int = 1,
) -> tuple[tuple[int, ...], ...]:
    """Construct a possibly rectangular cyclic block.

    Row ``i`` is obtained from the base pattern by a cyclic shift of
    ``i * shift``. For a square block this is the usual circulant matrix whose
    first column is ``pattern``. A rectangular block is a constrained extension
    that remains useful when the vertex partition has unequal block sizes.
    """

    if row_count < 1 or column_count < 1:
        raise ValueError("block dimensions must be positive")
    _validate_binary_pattern(pattern, column_count, "pattern")
    return tuple(
        tuple(pattern[(row * shift - column) % column_count] for column in range(column_count))
        for row in range(row_count)
    )


def build_block_cyclic_matrix(
    block_sizes: Sequence[int],
    diagonal_columns: Sequence[Sequence[int]],
    cross_patterns: Mapping[BlockPosition, Sequence[int]],
    *,
    cross_shifts: Mapping[BlockPosition, int] | None = None,
) -> AdjacencyMatrix:
    """Build a symmetric block matrix from cyclic diagonal and cross blocks.

    ``cross_patterns[(i, j)]`` is required for every ``i < j`` and has length
    ``block_sizes[j]``. The corresponding lower block is the transpose, so the
    resulting matrix always represents an undirected graph when each diagonal
    column is a valid simple symmetric circulant pattern.
    """

    sizes = tuple(block_sizes)
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("block sizes must be positive")
    if len(diagonal_columns) != len(sizes):
        raise ValueError("one diagonal column is required for every block")

    shifts = cross_shifts or {}
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    matrix = [[0] * offsets[-1] for _ in range(offsets[-1])]

    for block_index, (size, column) in enumerate(zip(sizes, diagonal_columns, strict=True)):
        _validate_binary_pattern(column, size, f"diagonal column {block_index}")
        diagonal_block = cyclic_block(size, size, column)
        start = offsets[block_index]
        for row in range(size):
            matrix[start + row][start : start + size] = diagonal_block[row]

    for first_block in range(len(sizes) - 1):
        for second_block in range(first_block + 1, len(sizes)):
            position = (first_block, second_block)
            if position not in cross_patterns:
                raise ValueError(f"missing cross pattern for blocks {position}")
            block = cyclic_block(
                sizes[first_block],
                sizes[second_block],
                cross_patterns[position],
                shift=shifts.get(position, 1),
            )
            first_start = offsets[first_block]
            second_start = offsets[second_block]
            for row in range(sizes[first_block]):
                for column in range(sizes[second_block]):
                    value = block[row][column]
                    matrix[first_start + row][second_start + column] = value
                    matrix[second_start + column][first_start + row] = value

    result = tuple(tuple(row) for row in matrix)
    validate_adjacency_matrix(result)
    return result


def two_block_search_dimension(first_size: int, second_size: int) -> int:
    """Return the number of free bits in the directed two-block family."""

    if first_size < 1 or second_size < 1:
        raise ValueError("block sizes must be positive")
    return first_size // 2 + second_size + second_size // 2


def decode_two_block_index(
    first_size: int, second_size: int, candidate_index: int
) -> TwoBlockParameters:
    """Decode a single search index into the three component indices."""

    dimension = two_block_search_dimension(first_size, second_size)
    if candidate_index < 0 or candidate_index >= 1 << dimension:
        raise ValueError(f"candidate_index must be in range(2**{dimension})")

    second_bits = second_size // 2
    cross_bits = second_size
    second_mask = (1 << second_bits) - 1
    cross_mask = (1 << cross_bits) - 1
    second_diagonal = candidate_index & second_mask
    cross_block = (candidate_index >> second_bits) & cross_mask
    first_diagonal = candidate_index >> (second_bits + cross_bits)
    return TwoBlockParameters(first_diagonal, cross_block, second_diagonal)


def build_two_block_candidate(
    first_size: int,
    second_size: int,
    candidate_index: int,
    *,
    cross_shift: int = 1,
) -> AdjacencyMatrix:
    """Build one candidate from the unequal two-block cyclic search family."""

    parameters = decode_two_block_index(first_size, second_size, candidate_index)
    first_column = symmetric_circulant_column(first_size, parameters.first_diagonal)
    second_column = symmetric_circulant_column(second_size, parameters.second_diagonal)
    cross_pattern = integer_to_pattern(second_size, parameters.cross_block)
    return build_block_cyclic_matrix(
        (first_size, second_size),
        (first_column, second_column),
        {(0, 1): cross_pattern},
        cross_shifts={(0, 1): cross_shift},
    )


def iter_two_block_candidates(
    first_size: int,
    second_size: int,
    *,
    start: int = 0,
    stop: int | None = None,
    cross_shift: int = 1,
) -> Iterator[tuple[int, AdjacencyMatrix]]:
    """Yield an index and matrix for a shard of the two-block search space."""

    total = 1 << two_block_search_dimension(first_size, second_size)
    upper = total if stop is None else stop
    if start < 0 or upper < start or upper > total:
        raise ValueError(f"expected 0 <= start <= stop <= {total}")
    for candidate_index in range(start, upper):
        yield candidate_index, build_two_block_candidate(
            first_size,
            second_size,
            candidate_index,
            cross_shift=cross_shift,
        )
