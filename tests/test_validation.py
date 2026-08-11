import tempfile
import unittest
from pathlib import Path

from book_ramsey.validation import (
    MatrixFormatError,
    parameters_from_filename,
    read_adjacency_matrix,
    verify_witness_file,
)

REPOSITORY_ROOT = Path(__file__).parents[1]
DECIDED_DIRECTORY = REPOSITORY_ROOT / "decidedRamseyNumber"

EXPECTED_WITNESSES = {
    "R(B3_B6_18).txt": (3, 6, 18),
    "R(B4_B5_18).txt": (4, 5, 18),
    "R(B5_B6_22).txt": (5, 6, 22),
    "R(B6_B7_26).txt": (6, 7, 26),
    "R(B6_B8_28).txt": (6, 8, 28),
    "R(B7_B8_30).txt": (7, 8, 30),
    "R(B8_B8_32).txt": (8, 8, 32),
    "R(B8_B9_34).txt": (8, 9, 34),
    "R(B9_B10_38).txt": (9, 10, 38),
    "R(B9_B11_40).txt": (9, 11, 40),
    "R(B10_B11_42).txt": (10, 11, 42),
    "R(B12_B13_50).txt": (12, 13, 50),
}


class DecidedMatrixTests(unittest.TestCase):
    def test_decided_matrices_are_valid_witnesses(self) -> None:
        for filename, parameters in EXPECTED_WITNESSES.items():
            with self.subTest(filename=filename):
                first_book, second_book, order = parameters
                result = verify_witness_file(DECIDED_DIRECTORY / filename)

                self.assertTrue(result.is_witness)
                self.assertEqual(result.order, order)
                self.assertEqual(result.first_book, first_book)
                self.assertEqual(result.second_book, second_book)
                self.assertLess(result.color_one_max, first_book)
                self.assertLess(result.color_zero_max, second_book)

    def test_parameters_are_read_from_filename(self) -> None:
        self.assertEqual(parameters_from_filename("R(B12_B13_50).txt"), (12, 13, 50))

    def test_rejects_a_non_symmetric_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.txt"
            path.write_text("010\n001\n000\n", encoding="utf-8")

            with self.assertRaisesRegex(MatrixFormatError, "differ"):
                read_adjacency_matrix(path)


if __name__ == "__main__":
    unittest.main()
