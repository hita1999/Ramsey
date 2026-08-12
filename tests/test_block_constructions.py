import unittest

from book_ramsey.block_constructions import (
    TwoBlockCyclicFactory,
    build_block_cyclic_matrix,
    build_two_block_candidate,
    cyclic_block,
    decode_two_block_index,
    integer_to_pattern,
    symmetric_circulant_column,
    two_block_search_dimension,
)
from book_ramsey.validation import validate_adjacency_matrix, verify_book_ramsey_witness


class BlockConstructionTests(unittest.TestCase):
    def test_two_block_factory_exposes_order_and_space_size(self) -> None:
        factory = TwoBlockCyclicFactory(13, 14)

        self.assertEqual(factory.order, 27)
        self.assertEqual(factory.space_size, 1 << 27)
        self.assertEqual(len(factory(0)), 27)

    def test_symmetric_circulant_column(self) -> None:
        column = symmetric_circulant_column(6, 0b101)
        self.assertEqual(column, (0, 1, 0, 1, 0, 1))
        matrix = cyclic_block(6, 6, column)
        validate_adjacency_matrix(matrix)

    def test_unequal_two_block_candidate_is_valid(self) -> None:
        matrix = build_two_block_candidate(10, 12, candidate_index=123456)
        self.assertEqual(len(matrix), 22)
        validate_adjacency_matrix(matrix)
        self.assertEqual(two_block_search_dimension(10, 12), 23)

    def test_two_block_index_decoding_covers_each_component(self) -> None:
        first_size = 10
        second_size = 12
        first = 0b10101
        cross = 0b101010101010
        second = 0b110011
        candidate = (first << (12 + 6)) | (cross << 6) | second
        parameters = decode_two_block_index(first_size, second_size, candidate)
        self.assertEqual(parameters.first_diagonal, first)
        self.assertEqual(parameters.cross_block, cross)
        self.assertEqual(parameters.second_diagonal, second)

    def test_equal_blocks_reproduce_known_b6_b8_witness(self) -> None:
        first_column = (0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0)
        cross_pattern = (0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1)
        second_column = (0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1)
        matrix = build_block_cyclic_matrix(
            (14, 14),
            (first_column, second_column),
            {(0, 1): cross_pattern},
        )
        result = verify_book_ramsey_witness(matrix, 6, 8)
        self.assertTrue(result.is_witness)
        self.assertEqual(result.order, 28)
        self.assertEqual(result.color_one_max, 5)
        self.assertEqual(result.color_zero_max, 7)

        # The legacy script printed the product 26 * 571 * 101 = 1499446,
        # which is not a unique search-space position. The new index is a
        # collision-free concatenation of the same three bit patterns.
        canonical_index = (26 << 21) | (571 << 7) | 101
        self.assertEqual(canonical_index, 54599141)
        self.assertEqual(TwoBlockCyclicFactory(14, 14)(canonical_index), matrix)

    def test_three_unequal_blocks_are_supported(self) -> None:
        sizes = (4, 5, 6)
        matrix = build_block_cyclic_matrix(
            sizes,
            tuple(symmetric_circulant_column(size, 0) for size in sizes),
            {
                (0, 1): integer_to_pattern(5, 0b10101),
                (0, 2): integer_to_pattern(6, 0b100101),
                (1, 2): integer_to_pattern(6, 0b011010),
            },
            cross_shifts={(0, 2): 2},
        )
        self.assertEqual(len(matrix), sum(sizes))
        validate_adjacency_matrix(matrix)


if __name__ == "__main__":
    unittest.main()
