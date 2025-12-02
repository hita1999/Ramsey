# Scripts

This directory groups together the Python scripts that were previously located in the repository root. They are organized by purpose through their filenames:

- `book_*` and `check_book*` scripts run Ramsey graph experiments related to book graphs.
- `generate_*`, `flip_matrix.py`, and `handmade_cirmatrix.py` handle matrix creation or manipulation for graph construction.
- `drawGraph.py`, `clique_in_book.py`, and other analysis helpers provide utilities for visualizing or inspecting graphs.
- `profiling.py`, `numslice_test.py`, and `test.py` capture small benchmarks or ad-hoc checks.

All scripts expect to be run from the repository root so their relative paths to data directories remain valid (for example: `python scripts/book_c1c2c3_multi.py`).
