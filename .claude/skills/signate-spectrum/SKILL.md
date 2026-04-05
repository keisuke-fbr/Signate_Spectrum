---
name: signate-spectrum
description: "近赤外研究会 スペクトル分析チャレンジ（SIGNATE）の実装支援スキル。近赤外スペクトルデータからの含水率予測タスクに関して、前処理・モデリング・バリデーション・提出ファイル作成を行う際に使用する。"
---

# SIGNATE スペクトル分析チャレンジ スキル

近赤外スペクトルデータを用いた木材含水率予測コンペの実装支援。

## Expertise Areas

- **EDA**: スペクトルデータの探索・可視化
- **Preprocessing**: ケモメトリクス的前処理（SNV, MSC, Savitzky-Golay 等）
- **Modeling**: PLS・機械学習・勾配ブースティングの実装
- **Validation**: 樹種をグループとした交差検証（GroupKFold）
- **Submission**: 提出ファイルの生成・スコア管理

## Reference Files

| Reference | When to Load |
|-----------|--------------|
| `references/competition-overview.md` | コンペ概要・スケジュール・賞金・提出ルール確認時 |
| `references/data-spec.md` | データ構造・列定義・樹種情報確認時 |
| `references/preprocessing.md` | 前処理手法の選択・実装時 |
| `references/modeling.md` | モデル選択・学習・ハイパーパラメータ調整時 |
| `references/validation.md` | バリデーション戦略・RMSE計算時 |

### Explicit Content Triggers

- 「前処理」「スペクトル」「SNV」「MSC」「Savitzky」→ `references/preprocessing.md`
- 「モデル」「LightGBM」「PLS」「SVR」→ `references/modeling.md`
- 「バリデーション」「CV」「GroupKFold」「RMSE」→ `references/validation.md`
- 「提出」「submission」「締切」→ `references/competition-overview.md`
- 「train」「test」「列」「樹種」「データ」→ `references/data-spec.md`
