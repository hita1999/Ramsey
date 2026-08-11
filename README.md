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

Python 3.9以降を想定しています。macOSではシステムに `python` コマンドがない場合があるため、最初の仮想環境作成には `python3` を使います。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[search,dev]'
```

検証CLIだけを使う場合は外部依存関係を必要としません。上と同様に仮想環境を作成・有効化してからインストールします。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
book-ramsey verify decidedRamseyNumber
```

ソースをインストールせずに実行する場合は次のとおりです。

```bash
PYTHONPATH=src python -m book_ramsey verify decidedRamseyNumber
```

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
