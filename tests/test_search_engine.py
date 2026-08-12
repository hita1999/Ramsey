import json
import tempfile
import unittest
from pathlib import Path

from book_ramsey.cli import main
from book_ramsey.search_engine import (
    EdgeBitsFactory,
    MatrixCandidateEvaluator,
    SearchConfig,
    run_search,
    score_book_candidate,
    write_search_result,
)


class NearMissScoreTests(unittest.TestCase):
    def test_counts_violating_spines_and_excess_pages(self) -> None:
        complete_triangle = (
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
        )

        score = score_book_candidate(complete_triangle, first_book=1, second_book=2)

        self.assertFalse(score.is_witness)
        self.assertEqual(score.color_one.maximum_pages, 1)
        self.assertEqual(score.color_one.violating_spines, 3)
        self.assertEqual(score.color_one.excess_pages, 3)
        self.assertEqual(score.color_one.maximum_excess, 1)
        self.assertEqual(score.color_zero.maximum_pages, 0)
        self.assertEqual(score.color_zero.violating_spines, 0)

    def test_edge_bits_factory_uses_lexicographic_upper_triangle(self) -> None:
        matrix = EdgeBitsFactory(order=3)(0b101)

        self.assertEqual(matrix, ((0, 1, 0), (1, 0, 1), (0, 1, 0)))


class SearchEngineTests(unittest.TestCase):
    def test_evaluator_rejects_a_factory_with_the_wrong_order(self) -> None:
        evaluator = MatrixCandidateEvaluator(
            EdgeBitsFactory(3), 1, 1, expected_order=4
        )

        with self.assertRaisesRegex(ValueError, "has order 3; expected 4"):
            evaluator(0)

    def test_search_finds_first_witness_and_tracks_best_candidate(self) -> None:
        evaluator = MatrixCandidateEvaluator(EdgeBitsFactory(3), 1, 1)
        outcome = run_search(
            evaluator,
            SearchConfig(first_book=1, second_book=1, order=3, start=0, stop=8),
        )

        self.assertTrue(outcome.found)
        self.assertEqual(outcome.witness.index, 1)
        self.assertEqual(outcome.best, outcome.witness)
        self.assertEqual(outcome.metadata.candidates_examined, 2)
        self.assertEqual(outcome.metadata.next_index, 2)

    def test_resume_continues_from_contiguous_checkpoint_prefix(self) -> None:
        evaluator = MatrixCandidateEvaluator(EdgeBitsFactory(3), 1, 1)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            first = run_search(
                evaluator,
                SearchConfig(
                    first_book=1,
                    second_book=1,
                    order=3,
                    start=0,
                    stop=1,
                    checkpoint_path=checkpoint,
                    checkpoint_every=1,
                    label="triangle",
                ),
            )
            resumed = run_search(
                evaluator,
                SearchConfig(
                    first_book=1,
                    second_book=1,
                    order=3,
                    start=0,
                    stop=8,
                    checkpoint_path=checkpoint,
                    resume=True,
                    label="triangle",
                ),
            )

            self.assertFalse(first.found)
            self.assertEqual(first.metadata.next_index, 1)
            self.assertEqual(resumed.metadata.effective_start, 1)
            self.assertEqual(resumed.witness.index, 1)
            self.assertEqual(resumed.metadata.candidates_examined, 2)

    def test_parallel_search_keeps_ordered_resume_boundary(self) -> None:
        evaluator = MatrixCandidateEvaluator(EdgeBitsFactory(3), 1, 1)
        try:
            outcome = run_search(
                evaluator,
                SearchConfig(
                    first_book=1,
                    second_book=1,
                    order=3,
                    start=0,
                    stop=8,
                    workers=2,
                    chunk_size=1,
                ),
            )
        except PermissionError as error:
            self.skipTest(f"process semaphores are unavailable: {error}")

        self.assertEqual(outcome.witness.index, 1)
        self.assertEqual(outcome.metadata.next_index, 2)

    def test_result_json_contains_reproducibility_metadata(self) -> None:
        outcome = run_search(
            MatrixCandidateEvaluator(EdgeBitsFactory(3), 1, 1),
            SearchConfig(first_book=1, second_book=1, order=3, start=0, stop=2),
        )
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "result.json"
            write_search_result(result_path, outcome)
            document = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertEqual(document["format_version"], 1)
        self.assertEqual(document["status"], "found")
        self.assertEqual(document["witness"]["index"], 1)
        self.assertIn("git_sha", document["metadata"])
        self.assertIn("elapsed_seconds", document["metadata"])

    def test_cli_can_write_result_checkpoint_and_witness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "result.json"
            checkpoint = root / "checkpoint.json"
            witness = root / "witness.txt"

            exit_code = main(
                [
                    "search",
                    "--first-book",
                    "1",
                    "--second-book",
                    "1",
                    "--order",
                    "3",
                    "--stop",
                    "8",
                    "--checkpoint",
                    str(checkpoint),
                    "--result-json",
                    str(result),
                    "--witness-output",
                    str(witness),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(result.is_file())
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(witness.read_text(encoding="utf-8"), "010\n100\n000\n")

    def test_cli_runs_unequal_two_block_factory(self) -> None:
        exit_code = main(
            [
                "search",
                "--first-book",
                "2",
                "--second-book",
                "2",
                "--order",
                "7",
                "--stop",
                "4",
                "--factory",
                "two-block:3:4",
            ]
        )

        self.assertIn(exit_code, (0, 1))


if __name__ == "__main__":
    unittest.main()
