# 出典台帳

## 一次資料

### 1. 現在の範囲 \(22\le R(B_4,B_7)\le23\)

- Bernard Lidický, Gwen McKinley, Florian Pfender, Steven Van Overberghe,
  “Small Ramsey Numbers for Books, Wheels, and Generalizations,”
  *The Electronic Journal of Combinatorics* 32(4) (2025), #P4.64.
  DOI: <https://doi.org/10.37236/13577>
  PDF: <https://www.combinatorics.org/ojs/index.php/eljc/article/download/v32i4p64/pdf/>
- 本研究との関係: Table 1 が \(R(B_4,B_7)\) の下界22、上界23を掲載する。論文の新上界は flag algebra によるが、本ディレクトリの上界23は Goodman 恒等式と強正則グラフの固有値だけで再証明した。
- 分類: 数値の既知状況は **文献依存**。本ノートの両側の証明は **証明済み**。

### 2. 動的サーベイ

- Stanisław P. Radziszowski,
  “Small Ramsey Numbers,”
  *The Electronic Journal of Combinatorics*, Dynamic Survey DS1, revision #18 (2026).
  PDF: <https://www.combinatorics.org/ojs/index.php/eljc/article/download/DS1/pdf/0>
- 本研究との関係: Book Ramsey 数の最新既知表を監査するために使用する。
- 分類: **文献依存**。

### 3. Book Ramsey 数の古典的結果

- Cecil C. Rousseau and John Sheehan,
  “On Ramsey numbers for books,”
  *Journal of Graph Theory* 2(1) (1978), 77--87.
  DOI: <https://doi.org/10.1002/jgt.3190020110>
- 本研究との関係: Book Ramsey 数の古典的な一般評価と小さい値の出典。\(KG(7,2)\) 型構成は \(R(B_4,B_6)\ge22\) にも使える。
- 分類: **文献依存**。

### 4. 代数的構成

- Lulu Dai and Qizhong Lin,
  “Book Ramsey numbers via algebraic constructions,”
  arXiv:2606.07214 (2026).
  <https://arxiv.org/abs/2606.07214>
- 本研究との関係: 三角グラフ \(T(7)\) とその補グラフによる下界構成の代数的文脈を確認する。同論文の \(R(B_{n-2},B_n)\) に関する定理は添字差3の \(R(B_4,B_7)\) には直接適用できない。
- 分類: **文献依存**。

## 本ノートで再証明した事項

以下は外部の計算結果を仮定せず、本ディレクトリ内に証明を置いた。

1. \(KG(7,2)\) による \(R(B_4,B_7)\ge22\)。
2. Goodman 恒等式。
3. Goodman 恒等式と \(SRG(23,10,3,5)\) の固有値矛盾による \(R(B_4,B_7)\le23\)。
4. 22頂点反例の不足量恒等式 \(3Q+2S=132\)。
5. 同恒等式の局所二重計数による独立検算。
6. 22頂点反例の次数範囲 \(5\le d_v\le13\)。
7. 境界次数5および13の局所構造と、両境界次数の排除。

## 引用方針

- 正確な値が未決定であること、既知表、発表年、計算手法は一次資料を引用する。
- 本ノート内で完結する組合せ論的導出には、主張ごとに証明を付す。
- 計算機で得た候補、探索ログ、未検証のパターンは証明済みの主張と混在させない。
