"""
submission.py
提出用スクリプト

モデル: LightGBM (num_leaves=15)
前処理: SNV + SG1d (window=7, polyorder=2)
波数帯域: wide_water (4500-5500 + 6500-7500 cm⁻¹)
評価: LOGO mean = 15.01
"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from lightgbm import LGBMRegressor
from datetime import datetime

# ============================================================
# データ読み込み
# ============================================================
print("📂 データ読み込み...")
train_df = pd.read_csv("../../data/train.csv", encoding="cp932")
test_df = pd.read_csv("../../data/test.csv", encoding="cp932")

target_col = "含水率"
meta_cols = ["sample number", "species number", "樹種", "含水率"]
spectrum_cols = [c for c in train_df.columns if c not in meta_cols]
wavenumbers = np.array([float(c) for c in spectrum_cols])

X_train_raw = train_df[spectrum_cols].to_numpy(dtype=float)
y_train = train_df[target_col].values
X_test_raw = test_df[spectrum_cols].to_numpy(dtype=float)

print(f"  訓練: {X_train_raw.shape}")
print(f"  テスト: {X_test_raw.shape}")

# ============================================================
# 波数帯域選択: wide_water (4500-5500 + 6500-7500 cm⁻¹)
# ============================================================
print("\n🔬 波数帯域選択: wide_water")

bands = [(4500, 5500), (6500, 7500)]
mask = np.zeros(len(wavenumbers), dtype=bool)
for lo, hi in bands:
    mask |= (wavenumbers >= lo) & (wavenumbers <= hi)

X_train_band = X_train_raw[:, mask]
X_test_band = X_test_raw[:, mask]
wn_selected = wavenumbers[mask]

print(f"  選択波数: {len(wn_selected)} / {len(wavenumbers)} ポイント")
print(f"  帯域: {bands}")

# ============================================================
# 前処理: SNV + SG1d (window=7)
# ============================================================
print("\n🔧 前処理: SNV + SG1d (window=7, polyorder=2)")

def snv(X):
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return (X - m) / s

X_train = savgol_filter(snv(X_train_band), window_length=7, polyorder=2, deriv=1, axis=1)
X_test = savgol_filter(snv(X_test_band), window_length=7, polyorder=2, deriv=1, axis=1)

print(f"  前処理後: 訓練 {X_train.shape}, テスト {X_test.shape}")

# ============================================================
# モデル学習: LightGBM (num_leaves=15)
# ============================================================
print("\n🚀 モデル学習: LightGBM (num_leaves=15)")

model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=15,
    verbosity=-1,
    random_state=42,
)
model.fit(X_train, y_train)

# ============================================================
# 予測
# ============================================================
pred = np.clip(model.predict(X_test).ravel(), 0, None)

print(f"\n📊 予測統計:")
print(f"  mean={pred.mean():.1f}, std={pred.std():.1f}")
print(f"  [{pred.min():.1f} ~ {pred.max():.1f}]")

print(f"\n  テスト樹種別:")
for sp in sorted(test_df["樹種"].unique()):
    mask_sp = test_df["樹種"] == sp
    v = pred[mask_sp]
    print(f"    {sp:12s}: mean={v.mean():6.1f} [{v.min():5.1f}~{v.max():5.1f}] n={mask_sp.sum()}")

# ============================================================
# 提出ファイル生成（ヘッダーなし）
# ============================================================
today = datetime.now().strftime("%Y%m%d")
filename = f"sub_{today}_LGB_nl15_SNV_SG1d_w7_wide_water.csv"

sub_df = pd.DataFrame({
    "sample number": test_df["sample number"],
    "含水率": pred,
})
sub_df.to_csv(filename, index=False, header=False, encoding="cp932")

print(f"\n✅ 提出ファイル: {filename}")
print(f"   ヘッダー: なし")
print(f"   行数: {len(sub_df)}")