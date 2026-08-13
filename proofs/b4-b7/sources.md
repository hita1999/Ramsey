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

### 5. Goodman恒等式から強正則性へのAI支援例

- Jeremy Kalfus and Bernard Lidický,
  “An automated proof that \(R(B_8,B_{10})=37\),”
  arXiv:2606.05629 (2026).
  <https://arxiv.org/abs/2606.05629>
- 本研究との関係: Goodman恒等式の等号条件から正則性・強正則性を導き、固有値で排除する近年の証明型を監査した。\(R(B_4,B_7)\) の22頂点問題では不足量が残るため、そのままでは強正則性に至らない。
- 分類: **文献依存**。

### 6. 最小固有値が (-2) 以上の正則グラフの分類

- Wenbin Wang and Yanli Zhu,
  “A note on regular graphs whose second largest eigenvalue does not exceed 1,”
  *Australasian Journal of Combinatorics* 94(2) (2026), 305--315.
  PDF: <https://ajc.maths.uq.edu.au/pdf/94/ajc_v94_p305.pdf>
- 本研究との関係: Theorem 1.1 と式 (1) が、最小固有値 (-2) 以上の連結正則グラフを、線グラフ、cocktail party graph、三層の exceptional graph に分類する既知結果を整理している。次数7の均一局所型で現れる14頂点6正則グラフについて、exceptional graph の三つの位数・次数関係はいずれも成立しないことを確認するために使う。
- 分類: 分類定理は **文献依存**。そこから二つの線グラフへ絞る計算は本ノートで **証明済み**。

## 本ノートで再証明した事項

以下は外部の計算結果を仮定せず、本ディレクトリ内に証明を置いた。

1. \(KG(7,2)\) による \(R(B_4,B_7)\ge22\)。
2. Goodman 恒等式。
3. Goodman 恒等式による \(R(B_4,B_6)=22\) と、22頂点反例における飽和青辺の存在。
4. Goodman 恒等式と \(SRG(23,10,3,5)\) の固有値矛盾による \(R(B_4,B_7)\le23\)。
5. 22頂点反例の不足量恒等式 \(3Q+2S=132\)。
6. 同恒等式の局所二重計数による独立検算。
7. 22頂点反例の第一次次数範囲 \(5\le d_v\le13\)。
8. 境界次数5および13の局所構造と、両境界次数の排除。
9. 赤・青局所グラフ間の集合族補題と、次数6および12の排除。
10. 第二次次数範囲 \(7\le d_v\le11\)。
11. 次数7頂点での青側完全飽和、領域別不足量、および \(\alpha\le11\)。
12. スペクトル分類後に残る二つの線グラフ局所型の手計算排除。
13. 次数7頂点どうしの隣接色と \(n_7\le2\)、非均一型から低不足量頂点・次数11頂点への伝播。

## 引用方針

- 正確な値が未決定であること、既知表、発表年、計算手法は一次資料を引用する。
- 本ノート内で完結する組合せ論的導出には、主張ごとに証明を付す。
- 計算機で得た候補、探索ログ、未検証のパターンは証明済みの主張と混在させない。
