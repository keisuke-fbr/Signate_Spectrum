# 近赤外研究会 スペクトル分析チャレンジ — 実装スキル集

## 1. コンペ概要

| 項目 | 内容 |
|------|------|
| タスク | 回帰（含水率予測） |
| 評価指標 | RMSE（小さいほど良い） |
| 締切 | 2026-06-30 23:59 |
| 提出上限 | 1日あたりの上限（要確認）、最終選択は最大2件 |
| 入賞対象 | 日本国内在住者のみ |
| 賞金 | 1位:10万, 2位:5万, 3位:3万（計18万円） |

---

## 2. データ仕様

### ターゲット変数
- `含水率 (%)` = (水の質量 / 全乾状態の木材質量) × 100

### 入力特徴量
- 近赤外スペクトル: **10000〜4000 cm⁻¹** の範囲
- 分解能: 8 cm⁻¹、32 scan 積算
- 列数: 約1555列（スペクトル波数列）

### データファイル
```
data/raw/train.csv   # 1322サンプル × 1559列（含水率あり）
data/raw/test.csv    # 550サンプル × 1558列（含水率なし）
```

### 列構成
| 列名 | 説明 |
|------|------|
| `sample number` | サンプルID |
| `species number` | 樹種番号 |
| `樹種` | 樹種名（日本語） |
| `含水率` | ターゲット（trainのみ） |
| `9993.76781` 〜 `3999.82139` | スペクトル吸光度値 |

### 樹種情報
- **train**: 13樹種（イチョウ、ウエンジ、ウォールナット、クリ、スプルース、チェリー、トチ、ナラ、ヒノキ、ベイスギ、米ヒバ、ベイマツ、ホワイトオーク）
- **test**: 6樹種（スギ、チーク、クスノキ、ケヤキ、ヤマザクラ、タモ）
- **注意**: trainとtestで樹種が完全に異なる → 樹種に依存しない汎化性能が必要

---

## 3. 評価指標

```python
from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

- リーダーボードは暫定評価（一部データ）→ 終了後に最終評価（残りデータ）へ切替
- 最終順位は精度だけでなく「前処理・モデリングの妥当性、再現性、解釈性」も審査対象

---

## 4. 実装上の重要ポイント

### 4.1 ドメイン知識（近赤外分光法の特性）
- 吸収帯がブロードで散乱・ベースライン変動の影響を強く受ける
- 試料状態（乾燥収縮によるプローブ距離変化）がスペクトルに影響
- 板目・まさ目・追いまさ面の違いが含まれる

### 4.2 前処理の候補
```python
# スペクトル前処理（ケモメトリクス的アプローチ）
# 1. 標準化 / 正規化
# 2. SNV (Standard Normal Variate)
# 3. MSC (Multiplicative Scatter Correction)
# 4. 1次・2次微分（Savitzky-Golay）
# 5. ベースライン補正

from scipy.signal import savgol_filter
sg = savgol_filter(X, window_length=11, polyorder=2, deriv=1)
```

### 4.3 モデルの候補
| アプローチ | 手法 |
|-----------|------|
| ケモメトリクス | PLS (Partial Least Squares) |
| 機械学習 | Ridge, Lasso, SVR, Random Forest |
| 勾配ブースティング | LightGBM, XGBoost, CatBoost |
| ニューラルネット | 1D-CNN, MLP |

### 4.4 バリデーション戦略
- **trainとtestで樹種が完全に異なる**ため、通常のランダムCV は信頼性が低い
- 樹種単位のLeave-One-Species-Out CV（LOSOCV）を推奨
```python
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=len(df['樹種'].unique()))
for train_idx, val_idx in gkf.split(X, y, groups=df['樹種']):
    ...
```

### 4.5 提出ファイル形式
```python
# 提出ファイル例（要確認）
submission = pd.DataFrame({
    'sample number': test_df['sample number'],
    '含水率': y_pred
})
submission.to_csv('submission.csv', index=False)
```

---

## 5. ファイル構成

```
Signate_Spectrum/
├── data/
│   └── raw/
│       ├── train.csv
│       └── test.csv
├── EDA/
│   ├── EDA.ipynb           # trainデータのEDA
│   └── EDA_test_data.ipynb # testデータのEDA
├── functions/              # 共通関数
├── main.ipynb              # メイン実装
└── Docs/
    └── competition_skills.md  # 本ファイル
```

---

## 6. スケジュール

| 日程 | イベント |
|------|---------|
| 2026-03-02 | コンペ開始 |
| 2026-05-31 | チーム作成期限 |
| 2026-06-30 | コンペ終了・提出締切 |
| 2026-07-10 | ソースコード等提出締切 |
| 2026-07中旬 | 幹事会審査 |
| 2026-08中旬 | 入賞者決定 |
| 2026-11-12 | 表彰式（ANS2026, 沖縄） |

---

## 7. 入賞時の提出物（コード品質に関わる要件）

- 学習・前処理のソースコード（再現可能であること）
- ソースコードの説明書（前処理・学習の流れが明記されていること）
- 実行環境（OS、ライブラリバージョン等）
- データ解釈・工夫点・モデリングから得られた示唆の説明資料
- 生成AI利用の有無・用途の申告
