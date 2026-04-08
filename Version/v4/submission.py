"""
submission_code1.py
Code 1 の分析結果を反映した提出スクリプト

改善点:
  ① ベイスギ除外
  ② log1p変換
  ③ ROI選択（1350-1650nm + 1900-2300nm）
  ④ 複数前処理結合（SNV + 1次微分 + 2次微分ROI）
  ⑤ Optunaハイパーパラメータ最適化
"""

import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. データ読み込み
# ============================================================
print("📂 データ読み込み...")
train_df = pd.read_csv("../../data/train.csv", encoding="cp932")
test_df  = pd.read_csv("../../data/test.csv",  encoding="cp932")

target_col    = "含水率"
meta_cols     = ["sample number", "species number", "樹種", "含水率"]
spectrum_cols = [c for c in train_df.columns if c not in meta_cols]
wavenumbers   = np.array([float(c) for c in spectrum_cols])
wavelengths   = 10_000_000 / wavenumbers  # nm

print(f"  訓練: {train_df.shape}")
print(f"  テスト: {test_df.shape}")

# ============================================================
# 2. ベイスギ除外
# ============================================================
print("\n🔧 ① ベイスギを訓練データから除外")
train_clean = train_df[train_df['樹種'] != 'ベイスギ'].reset_index(drop=True)
print(f"  除外前: {len(train_df)} → 除外後: {len(train_clean)} "
      f"(ベイスギ {len(train_df) - len(train_clean)} サンプル除外)")

X_train_raw    = train_clean[spectrum_cols].to_numpy(dtype=float)
y_train_raw    = train_clean[target_col].values
species_train  = train_clean["species number"].values
X_test_raw     = test_df[spectrum_cols].to_numpy(dtype=float)

# ============================================================
# 3. log1p 変換
# ============================================================
print("🔧 ② 含水率を log1p 変換")
y_train_log = np.log1p(y_train_raw)
print(f"  変換前: [{y_train_raw.min():.1f} ~ {y_train_raw.max():.1f}]")
print(f"  変換後: [{y_train_log.min():.2f} ~ {y_train_log.max():.2f}]")

# ============================================================
# 4. ROI 選択 + 複数前処理結合
# ============================================================
print("🔧 ③ ROI 選択（1350-1650nm + 1900-2300nm）")
roi_mask = (
    ((wavelengths >= 1350) & (wavelengths <= 1650)) |
    ((wavelengths >= 1900) & (wavelengths <= 2300))
)
roi_cols_idx = np.where(roi_mask)[0]
print(f"  選択波数: {len(roi_cols_idx)} / {len(wavenumbers)}")

print("🔧 ④ 複数前処理結合（SNV + 1次微分 + 2次微分ROI）")

def preprocess(X_raw, roi_idx):
    """各サンプル独立に処理（リーク無し）"""
    # SNV
    m = X_raw.mean(axis=1, keepdims=True)
    s = X_raw.std(axis=1, keepdims=True)
    X_snv = (X_raw - m) / (s + 1e-8)

    # 1次微分（全波数）
    X_d1 = savgol_filter(X_snv, window_length=15, polyorder=2, deriv=1, axis=1)

    # 2次微分（ROI のみ）
    X_d2_roi = savgol_filter(
        X_snv[:, roi_idx], window_length=15, polyorder=2, deriv=2, axis=1
    )

    return np.hstack([X_snv, X_d1, X_d2_roi])

X_train = preprocess(X_train_raw, roi_cols_idx)
X_test  = preprocess(X_test_raw,  roi_cols_idx)
print(f"  特徴量数: {X_train.shape[1]}  "
      f"(SNV:{X_train_raw.shape[1]} + D1:{X_train_raw.shape[1]} + D2roi:{len(roi_cols_idx)})")

# ============================================================
# 5. Optuna 最適化（GroupKFold 3 分割）
# ============================================================
print("\n" + "=" * 70)
print("🚀 ⑤ Optuna 最適化（GroupKFold 3 分割, 50 trials）")
print("=" * 70)

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    param = {
        'objective':        'regression',
        'metric':           'rmse',
        'verbosity':        -1,
        'random_state':     42,
        'n_jobs':           -1,
        'n_estimators':     1000,
        'learning_rate':    trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
        'max_depth':        trial.suggest_int('max_depth', 3, 7),
        'num_leaves':       trial.suggest_int('num_leaves', 8, 50),
        'subsample':        trial.suggest_float('subsample', 0.5, 0.9),
        'subsample_freq':   1,
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.05, 0.3),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 40),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
    }

    gkf = GroupKFold(n_splits=3)
    rmses = []
    for tr_idx, va_idx in gkf.split(X_train, y_train_log, groups=species_train):
        model = lgb.LGBMRegressor(**param)
        model.fit(
            X_train[tr_idx], y_train_log[tr_idx],
            eval_set=[(X_train[va_idx], y_train_log[va_idx])],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        preds  = np.expm1(model.predict(X_train[va_idx]))
        actual = np.expm1(y_train_log[va_idx])
        rmses.append(np.sqrt(mean_squared_error(actual, preds)))
    return np.mean(rmses)

pbar = tqdm(total=50, desc="🔍 Optuna", bar_format='{l_bar}{bar:30}{r_bar}')
best_so_far = float('inf')

def optuna_callback(study, trial):
    global best_so_far
    pbar.update(1)
    if trial.value < best_so_far:
        best_so_far = trial.value
        pbar.set_postfix_str(f"Best={best_so_far:.3f} (trial {trial.number})")

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, callbacks=[optuna_callback])
pbar.close()

print(f"\n🏆 Best CV RMSE: {study.best_value:.4f}")
print("  パラメータ:")
for k, v in study.best_params.items():
    print(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v}")

# ============================================================
# 6. LOGO 評価（最終確認）
# ============================================================
print("\n" + "=" * 70)
print("🔍 LOGO 評価（最終パラメータ）")
print("=" * 70)

best_params = study.best_params.copy()
best_params.update({
    'objective':    'regression',
    'metric':       'rmse',
    'verbosity':    -1,
    'random_state': 42,
    'n_jobs':       -1,
    'n_estimators': 1000,
})

sp_names_map = {}
for sp_num in np.unique(species_train):
    sp_names_map[sp_num] = train_clean.loc[
        train_clean['species number'] == sp_num, '樹種'
    ].iloc[0]

logo = LeaveOneGroupOut()
logo_rmses = {}
for tr_idx, va_idx in logo.split(X_train, y_train_log, groups=species_train):
    sp = species_train[va_idx][0]
    m = lgb.LGBMRegressor(**best_params)
    m.fit(
        X_train[tr_idx], y_train_log[tr_idx],
        eval_set=[(X_train[va_idx], y_train_log[va_idx])],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    preds  = np.expm1(m.predict(X_train[va_idx]))
    actual = np.expm1(y_train_log[va_idx])
    logo_rmses[sp] = np.sqrt(mean_squared_error(actual, preds))

rmse_vals = list(logo_rmses.values())
logo_mean   = np.mean(rmse_vals)
logo_median = np.median(rmse_vals)
print(f"  LOGO mean: {logo_mean:.2f},  median: {logo_median:.2f}")
for sp_num, rmse in sorted(logo_rmses.items(), key=lambda x: x[1], reverse=True):
    marker = "🔴" if rmse > 30 else "🟡" if rmse > 20 else "🟢"
    print(f"    {marker} {sp_names_map[sp_num]:15s} RMSE={rmse:6.1f}")

# ============================================================
# 7. 全訓練データで最終モデル学習 → テスト予測
# ============================================================
print("\n" + "=" * 70)
print("🎯 最終モデル学習 → テスト予測")
print("=" * 70)

final_model = lgb.LGBMRegressor(**best_params)
final_model.fit(X_train, y_train_log)

pred_log = final_model.predict(X_test)
pred     = np.clip(np.expm1(pred_log), 0, None)

print(f"\n📊 予測統計:")
print(f"  mean={pred.mean():.1f}, std={pred.std():.1f}")
print(f"  [{pred.min():.1f} ~ {pred.max():.1f}]")

print(f"\n  テスト樹種別:")
for sp in sorted(test_df["樹種"].unique()):
    mask_sp = test_df["樹種"] == sp
    v = pred[mask_sp]
    print(f"    {sp:12s}: mean={v.mean():6.1f} "
          f"[{v.min():5.1f}~{v.max():5.1f}] n={mask_sp.sum()}")

# ============================================================
# 8. 提出ファイル生成（ヘッダーなし）
# ============================================================
today    = datetime.now().strftime("%Y%m%d")
filename = f"sub_{today}_code1_LGB_optuna_log1p_noBeisugi_ROI.csv"

sub_df = pd.DataFrame({
    "sample number": test_df["sample number"],
    "含水率": pred,
})
sub_df.to_csv(filename, index=False, header=False, encoding="cp932")

print(f"""
{'='*70}
✅ 提出ファイル生成完了
{'='*70}
  ファイル名 : {filename}
  ヘッダー   : なし
  行数       : {len(sub_df)}
  LOGO mean  : {logo_mean:.2f}
  
  適用した工夫:
    ① ベイスギ除外
    ② log1p 変換
    ③ ROI 選択 (1350-1650nm + 1900-2300nm)
    ④ 複数前処理結合 (SNV + D1 + D2roi)
    ⑤ Optuna 最適化 (50 trials)
{'='*70}
""")