# \(R(B_4,B_7)\) 証明型研究

このディレクトリは、Book Ramsey 数 \(R(B_4,B_7)\) を計算機全探索に依存せず決定するための研究記録である。

## 現在の到達点

| 主張 | 状態 | 場所 |
|---|---|---|
| \(R(B_4,B_7)\ge 22\) | 証明済み | [`current-bounds.md`](current-bounds.md) |
| \(R(B_4,B_7)\le 23\) | 証明済み | [`current-bounds.md`](current-bounds.md) |
| \(R(B_4,B_6)=22\) と飽和青辺の存在 | 証明済み | [`current-bounds.md`](current-bounds.md) |
| \(3\sum_v(d_v-10)^2+2S=132\) | 証明済み | [`research-notes.md`](research-notes.md) |
| 22頂点反例では \(7\le d_v\le11\) | 証明済み | [`research-notes.md`](research-notes.md) |
| 次数5・6・12・13の反例候補 | 反証済み | [`research-notes.md`](research-notes.md) |
| 赤・青局所間の集合族補題 | 証明済み | [`research-notes.md`](research-notes.md) |
| 次数7頂点の青側完全分類 | 証明済み | [`research-notes.md`](research-notes.md) |
| 次数7頂点は高々2個 | 証明済み | [`research-notes.md`](research-notes.md) |
| 次数7頂点が二つある分岐 | 反証済み | [`n7-two-projection.md`](n7-two-projection.md) |
| 次数11頂点の赤局所不足量は1または3 | 証明済み | [`research-notes.md`](research-notes.md) |
| 次数7の均一局所型 \(r=0,q_u=3\) | 反証済み | [`research-notes.md`](research-notes.md) |
| 次数7・\(\alpha=1\) の分岐 | 6次Gram行列と根付き三正則小型から反証済み | [`n7-alpha1-gram.md`](n7-alpha1-gram.md) |
| 次数7・\(\alpha=3\) の分岐 | 三角形排除、6次集合Gram、\(N\times O\) 二重計数から反証済み | [`n7-alpha3-gram.md`](n7-alpha3-gram.md) |
| 次数7・\(\alpha=5\) の分岐 | 五つの次数11局所型、6次集合Gram、重み付き次数和から反証済み | [`n7-alpha5-gram.md`](n7-alpha5-gram.md) |
| 次数7・\(\alpha=7,9,11\) の分岐 | 正規化不足量と共通の五標準型から一括反証済み | [`n7-alpha7-11-gram.md`](n7-alpha7-11-gram.md) |
| 22頂点反例の最小次数 | \(\delta(G)\ge8\) | [`n7-alpha7-11-gram.md`](n7-alpha7-11-gram.md) |
| 22頂点反例では飽和赤辺44本以上・飽和青辺55本以上 | 証明済み | [`research-notes.md`](research-notes.md) |
| 22頂点反例は存在しない | 予想 | 未決定 |
| 22頂点反例は存在する | 予想 | 上と排他的な未決定候補 |

文献で知られている範囲は

\[
22\le R(B_4,B_7)\le23
\]

である。このディレクトリでは、その両側を自足的に証明したうえで、残る22頂点の場合に必要な条件を手計算で絞り込む。

## ファイル

- [`current-bounds.md`](current-bounds.md): 既知範囲 \(22\le R(B_4,B_7)\le23\) の整形済み非計算証明。
- [`research-notes.md`](research-notes.md): 22頂点反例を仮定した恒等式、局所補題、次の研究課題。
- [`n7-two-projection.md`](n7-two-projection.md): 次数7頂点が二つある分岐を射影不等式と有理線形証明書で排除する自足的証明。
- [`n7-alpha1-gram.md`](n7-alpha1-gram.md): 唯一の次数7中心で赤局所不足量 \(\alpha=1\) となる分岐を6次Gram行列で排除する自足的証明。
- [`n7-alpha3-gram.md`](n7-alpha3-gram.md): 唯一の次数7中心で赤局所不足量 \(\alpha=3\) となる分岐を三つの次数11局所型と6次集合Gramから排除する自足的証明。
- [`n7-alpha5-gram.md`](n7-alpha5-gram.md): 唯一の次数7中心で赤局所不足量 \(\alpha=5\) となる分岐を五つの次数11局所型と6次集合Gramから排除する自足的証明。
- [`n7-alpha7-11-gram.md`](n7-alpha7-11-gram.md): 残る \(\alpha=7,9,11\) を正規化し、同じ五標準型へ帰着して次数7中心を完全排除する証明。
- [`sources.md`](sources.md): 出典、各文献への依存関係、確認状況。
- [`verification.md`](verification.md): 独立再導出、機械検算、Git差分監査の記録。

## 状態ラベル

- **証明済み**: このディレクトリ内に自足的な証明がある。
- **予想**: 証明されていない研究上の仮説。
- **反証済み**: 反例または矛盾が示されている。
- **文献依存**: 本ノート内では再証明せず、出典に依存する。

## 研究上の制約

- 計算機は恒等式の検算、候補次数列の列挙、誤予想の早期排除に限って用いる。
- 最終的な決定証明はコード、SAT/SMT、有限全探索、浮動小数点計算に依存させない。
- 完全証明が得られるまでは Markdown で研究し、得られた段階で日本語 LaTeX 原稿へ移す。
- Python パッケージ、CLI、探索 API には変更を加えない。

## Git運用

この研究は `main` 起点の `feature/b4-b7-proof` と専用worktreeで行う。`main` へのマージ、PRのマージ、直接pushは、明示的な承認を得るまで行わない。
