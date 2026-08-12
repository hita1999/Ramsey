# Book Ramsey search

Bookグラフの2色Ramsey数に対する下界構成を、循環行列とブロック循環行列を用いて探索する研究リポジトリです。2024年の論文で記述した探索法をPythonで実装し、探索中間物と検証済みの隣接行列を保存しています。

## 主な結果

`decidedRamseyNumber/` には、既知の上界と一致してRamsey数を決定する下界構成が入っています。ファイル名の末尾はグラフの位数、したがってRamsey数から1を引いた値です。

| Ramsey数 | 臨界グラフの位数 |
| --- | ---: |
| $R(B_3,B_6)=19$ | 18 |
| $R(B_4,B_5)=19$ | 18 |
| $R(B_5,B_6)=23$ | 22 |
| $R(B_6,B_7)=27$ | 26 |
| $R(B_6,B_8)=29$ | 28 |
| $R(B_7,B_8)=31$ | 30 |
| $R(B_8,B_8)=33$ | 32 |
| $R(B_8,B_9)=35$ | 34 |
| $R(B_9,B_{10})=39$ | 38 |
| $R(B_9,B_{11})=41$ | 40 |
| $R(B_{10},B_{11})=43$ | 42 |
| $R(B_{12},B_{13})=51$ | 50 |

## 探索法

完全グラフの2-coloringを0/1隣接行列として表し、次の順に探索空間を広げます。

1. 1個のcoloringベクトルで定まる対称循環行列
2. 対角・非対角ブロックが対称循環行列であるブロック循環行列
3. 非対角ブロックに非対称循環行列を許したブロック循環行列
4. 3ブロック、4ブロックへの拡張

辺 $uv$ を背表紙とする単色bookのページ数は、その色における $u,v$ の共通近傍数です。すべての色1の辺で共通近傍数が $m$ 未満、すべての色0の辺で共通近傍数が $n$ 未満なら、その行列は $R(B_m,B_n)$ の下界構成になります。

論文と既存スクリプトの対応は次のとおりです。

| 探索空間 | 主なスクリプト |
| --- | --- |
| 循環行列 | `book_c1_multi.py` |
| 2ブロック・対称非対角 | `book_c1c2c3_multi.py` |
| 2ブロック・非対称非対角 | `book_c1c2c3_asymmetric.py` |
| 3ブロック | `book_c1toc6_multi.py`, `book_c1toc6_asymmetric.py` |
| 4ブロック | `book_c1toc10_multi.py` |

## セットアップ

Python 3.13以降を想定しています。最初に利用するPythonのバージョンを確認してください。

```bash
python3.13 --version
```

macOSではシステムの `python3` が古い場合があるため、仮想環境の作成時に
`python3.13` を明示します。

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[search,dev]'
```

検証CLIだけを使う場合は外部依存関係を必要としません。上と同様に仮想環境を作成・有効化してからインストールします。

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
book-ramsey verify decidedRamseyNumber
```

ソースをインストールせずに実行する場合は次のとおりです。

```bash
PYTHONPATH=src python -m book_ramsey verify decidedRamseyNumber
```

## 再利用可能な探索エンジン

`book-ramsey search` は候補番号の半開区間 `[start, stop)` を探索します。小さい問題の自己診断には、完全グラフの上三角を整数のビット列として列挙する組み込みの `edge-bits` 構成を利用できます。次の例は3頂点上で $B_1$ を両色とも避ける彩色を探します。

```bash
book-ramsey search \
  --first-book 1 --second-book 1 --order 3 \
  --start 0 --stop 8 --workers 2 --chunk-size 1 \
  --checkpoint runs/b1-b1.json \
  --result-json runs/b1-b1-result.json \
  --witness-output runs/b1-b1-matrix.txt
```

中断後は同じ条件に `--resume` を追加します。完了済みの連続区間の直後から再開するため、`--stop` を大きくして探索範囲を延長することもできます。

```bash
book-ramsey search \
  --first-book 1 --second-book 1 --order 3 \
  --start 0 --stop 8 \
  --checkpoint runs/b1-b1.json --resume
```

独自の構成は、候補番号を1個受け取り0/1隣接行列を返す呼び出し可能オブジェクトとして公開し、`--factory package.module:factory` で指定します。複数workerで使うfactoryはpickle可能かつ決定的である必要があります。ライブラリからは `MatrixCandidateEvaluator` と `run_search` を直接利用できます。

異なる大きさの2ブロック循環構成は組み込みfactoryで探索できます。たとえば22頂点を
10+12に分け、最初の100万候補を調べるコマンドは次のとおりです。

```bash
book-ramsey search \
  --first-book 4 --second-book 7 --order 22 \
  --start 0 --stop 1000000 --workers 4 \
  --factory two-block:10:12 \
  --checkpoint runs/b4-b7-10x12.json
```

各候補について、色ごとの最大book、禁止サイズ以上のbookを持つ背表紙数、許容上限を超えたページ数の合計・最大値を記録します。候補の比較は「違反背表紙数、超過ページ量、最大超過量」の順で行うため、解がない探索でも最良のnear-missを次の局所探索に渡せます。結果JSONには探索範囲、検査候補数、経過時間、worker数、Python・OS情報、Git SHAと未コミット変更の有無も保存されます。

## ディレクトリ構成

| パス | 内容 |
| --- | --- |
| `src/book_ramsey/` | 再利用可能な検証コードとCLI |
| `tests/` | 検証コードと決定済み行列のテスト |
| `decidedRamseyNumber/` | 検証済みの下界構成 |
| `generatedMatrix/` | 探索で生成した候補・成功構成 |
| `searchResultTextFile/` | 過去の生の探索ログ。約94MBあるため通常は読み込まない |
| `adjcencyMatrix/` | 探索の種にした既知グラフ。歴史的な綴りを互換性のため維持 |
| `targetAdjcencyMatrix/` | 検査用bookグラフ |
| `formerCheckMatrix/` | 初期の検証実装 |
| `other/` | Book Ramsey探索以外の実験 |

ルート直下のスクリプトは、研究時の実験条件とコミット履歴を保つため現時点では移動していません。新しい共通処理は `src/book_ramsey/` に追加し、段階的に重複を解消します。

## 検証

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

検証器は隣接行列の正方性、0/1要素、対称性、対角成分、および両色の最大bookサイズを独立に確認します。
