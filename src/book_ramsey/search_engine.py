"""Reusable, dependency-free exhaustive search support.

The historical search scripts in the repository combine construction, parallel
execution, progress reporting, and result persistence.  This module keeps those
concerns separate: a deterministic candidate factory maps an integer to an
adjacency matrix, while :func:`run_search` owns range handling and persistence.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .bitset import BitsetGraph
from .validation import AdjacencyMatrix

CHECKPOINT_VERSION = 1


class CandidateEvaluator(Protocol):
    """A pickleable callable that evaluates one integer-indexed candidate."""

    def __call__(self, index: int) -> CandidateEvaluation: ...


@dataclass(frozen=True)
class ColorBookScore:
    """Book statistics for one color.

    ``excess_pages`` measures pages above the largest permitted book.  Thus a
    spine with ``target + 2`` pages contributes three, not two: reaching
    ``target`` already violates the condition.
    """

    target: int
    maximum_pages: int
    maximum_spine: tuple[int, int] | None
    violating_spines: int
    excess_pages: int
    maximum_excess: int


@dataclass(frozen=True)
class NearMissScore:
    """A deterministic score for comparing invalid Ramsey candidates."""

    color_one: ColorBookScore
    color_zero: ColorBookScore

    @property
    def is_witness(self) -> bool:
        return self.violating_spines == 0

    @property
    def violating_spines(self) -> int:
        return self.color_one.violating_spines + self.color_zero.violating_spines

    @property
    def excess_pages(self) -> int:
        return self.color_one.excess_pages + self.color_zero.excess_pages

    @property
    def maximum_excess(self) -> int:
        return max(self.color_one.maximum_excess, self.color_zero.maximum_excess)

    @property
    def rank(self) -> tuple[int, int, int, int, int]:
        """Sort key where a lower tuple denotes a better candidate."""

        return (
            self.violating_spines,
            self.excess_pages,
            self.maximum_excess,
            self.color_one.maximum_pages,
            self.color_zero.maximum_pages,
        )


@dataclass(frozen=True)
class CandidateEvaluation:
    """Evaluation of the candidate identified by ``index``."""

    index: int
    score: NearMissScore


@dataclass(frozen=True)
class MatrixCandidateEvaluator:
    """Adapt a matrix factory to the generic integer search engine."""

    factory: Callable[[int], Sequence[Sequence[int]]]
    first_book: int
    second_book: int
    expected_order: int | None = None

    def __call__(self, index: int) -> CandidateEvaluation:
        matrix = self.factory(index)
        if self.expected_order is not None and len(matrix) != self.expected_order:
            raise ValueError(
                f"candidate {index} has order {len(matrix)}; "
                f"expected {self.expected_order}"
            )
        return CandidateEvaluation(
            index=index,
            score=score_book_candidate(matrix, self.first_book, self.second_book),
        )


@dataclass(frozen=True)
class EdgeBitsFactory:
    """Decode an integer as the upper triangle of an unrestricted graph.

    This construction is intended for small correctness checks.  Bit zero is
    edge ``(0, 1)``, followed lexicographically by ``(0, 2)``, ``(0, 3)``, ...
    """

    order: int

    @property
    def space_size(self) -> int:
        return 1 << (self.order * (self.order - 1) // 2)

    def __call__(self, index: int) -> AdjacencyMatrix:
        if index < 0 or index >= self.space_size:
            raise ValueError(f"candidate index {index} is outside [0, {self.space_size})")
        matrix = [[0] * self.order for _ in range(self.order)]
        bit = 0
        for first in range(self.order - 1):
            for second in range(first + 1, self.order):
                color = (index >> bit) & 1
                matrix[first][second] = color
                matrix[second][first] = color
                bit += 1
        return tuple(tuple(row) for row in matrix)


@dataclass(frozen=True)
class SearchConfig:
    """Configuration for an exclusive-stop exhaustive search range."""

    first_book: int
    second_book: int
    order: int
    stop: int
    start: int = 0
    workers: int = 1
    chunk_size: int = 256
    checkpoint_path: Path | None = None
    checkpoint_every: int = 10_000
    resume: bool = False
    label: str = "search"
    repository: Path | None = None

    def validate(self) -> None:
        if self.first_book < 1 or self.second_book < 1:
            raise ValueError("book sizes must be positive")
        if self.order < 1:
            raise ValueError("order must be positive")
        if self.start < 0 or self.stop < self.start:
            raise ValueError("expected 0 <= start <= stop")
        if self.workers < 1:
            raise ValueError("workers must be positive")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be positive")
        if self.resume and self.checkpoint_path is None:
            raise ValueError("resume requires checkpoint_path")


@dataclass(frozen=True)
class SearchMetadata:
    """Reproducibility metadata captured for a search invocation."""

    label: str
    first_book: int
    second_book: int
    order: int
    requested_start: int
    requested_stop: int
    effective_start: int
    next_index: int
    candidates_examined: int
    elapsed_seconds: float
    workers: int
    chunk_size: int
    started_at: str
    finished_at: str
    git_sha: str | None
    git_dirty: bool | None
    python_version: str
    platform: str
    hostname: str


@dataclass(frozen=True)
class SearchOutcome:
    """Result and best observed near miss from an exhaustive range."""

    witness: CandidateEvaluation | None
    best: CandidateEvaluation | None
    metadata: SearchMetadata

    @property
    def found(self) -> bool:
        return self.witness is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": CHECKPOINT_VERSION,
            "status": "found" if self.found else "complete",
            "witness": _evaluation_to_dict(self.witness),
            "best": _evaluation_to_dict(self.best),
            "metadata": asdict(self.metadata),
        }


def _score_color(
    rows: tuple[int, ...], order: int, target: int
) -> ColorBookScore:
    maximum_pages = 0
    maximum_spine: tuple[int, int] | None = None
    violating_spines = 0
    excess_pages = 0
    maximum_excess = 0
    for first in range(order - 1):
        first_row = rows[first]
        for second in range(first + 1, order):
            if not first_row & (1 << second):
                continue
            pages = (first_row & rows[second]).bit_count()
            if maximum_spine is None or pages > maximum_pages:
                maximum_pages = pages
                maximum_spine = (first, second)
            excess = max(0, pages - target + 1)
            if excess:
                violating_spines += 1
                excess_pages += excess
                maximum_excess = max(maximum_excess, excess)

    return ColorBookScore(
        target=target,
        maximum_pages=maximum_pages,
        maximum_spine=maximum_spine,
        violating_spines=violating_spines,
        excess_pages=excess_pages,
        maximum_excess=maximum_excess,
    )


def score_book_candidate(
    matrix: Sequence[Sequence[int]], first_book: int, second_book: int
) -> NearMissScore:
    """Return witness status and near-miss statistics for a two-coloring."""

    if first_book < 1 or second_book < 1:
        raise ValueError("book sizes must be positive")
    graph = BitsetGraph.from_adjacency_matrix(matrix)
    return NearMissScore(
        color_one=_score_color(
            graph.color_one_rows, graph.order, target=first_book
        ),
        color_zero=_score_color(
            graph.color_zero_rows, graph.order, target=second_book
        ),
    )


def _git_state(repository: Path | None) -> tuple[str | None, bool | None]:
    working_directory = repository or Path.cwd()
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=working_directory,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=working_directory,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return sha, dirty


def _evaluation_to_dict(evaluation: CandidateEvaluation | None) -> dict[str, object] | None:
    return asdict(evaluation) if evaluation is not None else None


def _evaluation_from_dict(value: object) -> CandidateEvaluation | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("invalid evaluation in checkpoint")
    score_value = value["score"]
    if not isinstance(score_value, dict):
        raise ValueError("invalid score in checkpoint")

    def color_score(name: str) -> ColorBookScore:
        raw = score_value[name]
        if not isinstance(raw, dict):
            raise ValueError("invalid color score in checkpoint")
        spine = raw["maximum_spine"]
        return ColorBookScore(
            target=int(raw["target"]),
            maximum_pages=int(raw["maximum_pages"]),
            maximum_spine=tuple(spine) if spine is not None else None,
            violating_spines=int(raw["violating_spines"]),
            excess_pages=int(raw["excess_pages"]),
            maximum_excess=int(raw["maximum_excess"]),
        )

    return CandidateEvaluation(
        index=int(value["index"]),
        score=NearMissScore(
            color_one=color_score("color_one"),
            color_zero=color_score("color_zero"),
        ),
    )


def _checkpoint_signature(config: SearchConfig) -> dict[str, object]:
    return {
        "label": config.label,
        "first_book": config.first_book,
        "second_book": config.second_book,
        "order": config.order,
        "requested_start": config.start,
    }


def _write_json_atomic(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_search_result(path: str | Path, outcome: SearchOutcome) -> None:
    """Write a completed search result as stable, human-readable JSON."""

    _write_json_atomic(Path(path), outcome.to_dict())


def _write_checkpoint(
    config: SearchConfig,
    *,
    next_index: int,
    candidates_examined: int,
    elapsed_seconds: float,
    best: CandidateEvaluation | None,
    witness: CandidateEvaluation | None,
    complete: bool,
) -> None:
    if config.checkpoint_path is None:
        return
    document: dict[str, object] = {
        "format_version": CHECKPOINT_VERSION,
        "signature": _checkpoint_signature(config),
        "requested_stop": config.stop,
        "next_index": next_index,
        "candidates_examined": candidates_examined,
        "elapsed_seconds": elapsed_seconds,
        "best": _evaluation_to_dict(best),
        "witness": _evaluation_to_dict(witness),
        "complete": complete,
    }
    _write_json_atomic(config.checkpoint_path, document)


def _load_checkpoint(
    config: SearchConfig,
) -> tuple[int, int, float, CandidateEvaluation | None, CandidateEvaluation | None]:
    assert config.checkpoint_path is not None
    try:
        document = json.loads(config.checkpoint_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"checkpoint does not exist: {config.checkpoint_path}") from error
    if document.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint format")
    if document.get("signature") != _checkpoint_signature(config):
        raise ValueError("checkpoint parameters do not match this search")
    next_index = int(document["next_index"])
    if next_index < config.start or next_index > config.stop:
        raise ValueError("checkpoint next_index is outside the requested range")
    return (
        next_index,
        int(document["candidates_examined"]),
        float(document["elapsed_seconds"]),
        _evaluation_from_dict(document.get("best")),
        _evaluation_from_dict(document.get("witness")),
    )


def _is_better(candidate: CandidateEvaluation, best: CandidateEvaluation | None) -> bool:
    return best is None or (candidate.score.rank, candidate.index) < (best.score.rank, best.index)


def run_search(evaluator: CandidateEvaluator, config: SearchConfig) -> SearchOutcome:
    """Evaluate ``[start, stop)`` in order, optionally resuming a checkpoint.

    Evaluation is ordered even with multiple worker processes, so ``next_index``
    always identifies a contiguous completed prefix and is safe to resume.
    """

    config.validate()
    effective_start = config.start
    candidates_examined = 0
    previous_elapsed = 0.0
    best: CandidateEvaluation | None = None
    witness: CandidateEvaluation | None = None
    if config.resume:
        (
            effective_start,
            candidates_examined,
            previous_elapsed,
            best,
            witness,
        ) = _load_checkpoint(config)

    started = datetime.now(UTC)
    timer_started = time.perf_counter()
    next_index = effective_start
    last_checkpoint_count = candidates_examined
    git_sha, git_dirty = _git_state(config.repository)

    def accept(evaluation: CandidateEvaluation) -> bool:
        nonlocal best, candidates_examined, next_index
        if evaluation.index != next_index:
            raise ValueError(
                f"evaluator returned index {evaluation.index}; expected {next_index}"
            )
        candidates_examined += 1
        next_index += 1
        if _is_better(evaluation, best):
            best = evaluation
        return evaluation.score.is_witness

    try:
        if witness is None and next_index < config.stop:
            if config.workers == 1:
                while next_index < config.stop:
                    evaluation = evaluator(next_index)
                    if accept(evaluation):
                        witness = evaluation
                        break
                    if candidates_examined - last_checkpoint_count >= config.checkpoint_every:
                        elapsed = previous_elapsed + time.perf_counter() - timer_started
                        _write_checkpoint(
                            config,
                            next_index=next_index,
                            candidates_examined=candidates_examined,
                            elapsed_seconds=elapsed,
                            best=best,
                            witness=witness,
                            complete=False,
                        )
                        last_checkpoint_count = candidates_examined
            else:
                batch_size = config.workers * config.chunk_size
                with ProcessPoolExecutor(max_workers=config.workers) as executor:
                    while next_index < config.stop:
                        batch_stop = min(config.stop, next_index + batch_size)
                        indices = range(next_index, batch_stop)
                        for evaluation in executor.map(
                            evaluator, indices, chunksize=config.chunk_size
                        ):
                            if accept(evaluation):
                                witness = evaluation
                                break
                        elapsed = previous_elapsed + time.perf_counter() - timer_started
                        _write_checkpoint(
                            config,
                            next_index=next_index,
                            candidates_examined=candidates_examined,
                            elapsed_seconds=elapsed,
                            best=best,
                            witness=witness,
                            complete=False,
                        )
                        last_checkpoint_count = candidates_examined
                        if witness is not None:
                            break
    except BaseException:
        elapsed = previous_elapsed + time.perf_counter() - timer_started
        _write_checkpoint(
            config,
            next_index=next_index,
            candidates_examined=candidates_examined,
            elapsed_seconds=elapsed,
            best=best,
            witness=witness,
            complete=False,
        )
        raise

    elapsed = previous_elapsed + time.perf_counter() - timer_started
    finished = datetime.now(UTC)
    complete = witness is not None or next_index >= config.stop
    _write_checkpoint(
        config,
        next_index=next_index,
        candidates_examined=candidates_examined,
        elapsed_seconds=elapsed,
        best=best,
        witness=witness,
        complete=complete,
    )
    metadata = SearchMetadata(
        label=config.label,
        first_book=config.first_book,
        second_book=config.second_book,
        order=config.order,
        requested_start=config.start,
        requested_stop=config.stop,
        effective_start=effective_start,
        next_index=next_index,
        candidates_examined=candidates_examined,
        elapsed_seconds=elapsed,
        workers=config.workers,
        chunk_size=config.chunk_size,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        git_sha=git_sha,
        git_dirty=git_dirty,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        hostname=socket.gethostname(),
    )
    return SearchOutcome(witness=witness, best=best, metadata=metadata)
