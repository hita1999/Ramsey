import unittest
from pathlib import Path

from book_ramsey.bitset import (
    BitsetGraph,
    adjacency_rows_to_bitsets,
    find_book_violation_bitset,
    maximum_book_size_bitset,
    verify_book_ramsey_witness_bitset,
)
from book_ramsey.validation import (
    parameters_from_filename,
    read_adjacency_matrix,
    verify_book_ramsey_witness,
)

REPOSITORY_ROOT = Path(__file__).parents[1]
DECIDED_DIRECTORY = REPOSITORY_ROOT / "decidedRamseyNumber"
DECIDED_WITNESSES = sorted(DECIDED_DIRECTORY.glob("R(B*_B*_*).txt"))


class BitsetValidationTests(unittest.TestCase):
    def test_matches_reference_for_every_decided_witness(self) -> None:
        self.assertEqual(len(DECIDED_WITNESSES), 12)

        for path in DECIDED_WITNESSES:
            with self.subTest(filename=path.name):
                first_book, second_book, _ = parameters_from_filename(path)
                matrix = read_adjacency_matrix(path)
                reference = verify_book_ramsey_witness(matrix, first_book, second_book)
                bitset = verify_book_ramsey_witness_bitset(
                    matrix, first_book, second_book
                )

                self.assertEqual(bitset, reference)
                self.assertTrue(bitset.is_witness)

    def test_builds_both_color_neighborhoods(self) -> None:
        matrix = (
            (0, 1, 0),
            (1, 0, 1),
            (0, 1, 0),
        )

        color_one, color_zero = adjacency_rows_to_bitsets(matrix)

        self.assertEqual(color_one, (0b010, 0b101, 0b010))
        self.assertEqual(color_zero, (0b100, 0b000, 0b001))

    def test_reports_maximum_and_first_violation_with_spines(self) -> None:
        complete_four = tuple(
            tuple(int(first != second) for second in range(4))
            for first in range(4)
        )
        graph = BitsetGraph.from_adjacency_matrix(complete_four)

        self.assertEqual(maximum_book_size_bitset(graph, color=1), (2, (0, 1)))
        violation = find_book_violation_bitset(graph, color=1, book_size=2)
        self.assertIsNotNone(violation)
        assert violation is not None
        self.assertEqual(violation.spine, (0, 1))
        self.assertEqual(violation.actual_pages, 2)
        self.assertEqual(violation.required_pages, 2)
        self.assertIsNone(find_book_violation_bitset(graph, color=0, book_size=1))

    def test_rejects_invalid_parameters(self) -> None:
        matrix = ((0, 1), (1, 0))
        graph = BitsetGraph.from_adjacency_matrix(matrix)

        with self.assertRaisesRegex(ValueError, "color must be 0 or 1"):
            maximum_book_size_bitset(graph, color=2)
        with self.assertRaisesRegex(ValueError, "book size must be positive"):
            find_book_violation_bitset(graph, color=1, book_size=0)
        with self.assertRaisesRegex(ValueError, "book sizes must be positive"):
            verify_book_ramsey_witness_bitset(matrix, first_book=0, second_book=1)


if __name__ == "__main__":
    unittest.main()
