# Scripts

This directory groups together the Python scripts that were previously located in the repository root. They are organized by purpose through their filenames:

- `book_*` and `check_book*` scripts run Ramsey graph experiments related to book graphs.
- `generate_*`, `flip_matrix.py`, and `handmade_cirmatrix.py` handle matrix creation or manipulation for graph construction.
- `drawGraph.py`, `clique_in_book.py`, and other analysis helpers provide utilities for visualizing or inspecting graphs.
- `profiling.py`, `numslice_test.py`, and `test.py` capture small benchmarks or ad-hoc checks.

All scripts expect to be run from the repository root so their relative paths to data directories remain valid (for example: `python scripts/book_c1c2c3_multi.py`).

---

## 日本語の説明 / Description in Japanese

このディレクトリには、以前リポジトリのルートに置かれていた Python スクリプトをまとめています。ファイル名のルールで役割を判別できます。

- `book_*` および `check_book*` 系のスクリプトは、ブックグラフに関連するラムゼーグラフの実験を実行します。
- `generate_*`、`flip_matrix.py`、`handmade_cirmatrix.py` は、グラフ構築のための行列生成・変換を担当します。
- `drawGraph.py`、`clique_in_book.py` などの解析補助スクリプトは、グラフの可視化や検査に利用できます。
- `profiling.py`、`numslice_test.py`、`test.py` は、小規模なベンチマークやアドホックな確認のためのスクリプトです。

各スクリプトはリポジトリのルートディレクトリで実行することを想定しています。データディレクトリへの相対パスが正しく解決されるよう、`python scripts/book_c1c2c3_multi.py` のようにプロジェクトのルートでコマンドを実行してください。
