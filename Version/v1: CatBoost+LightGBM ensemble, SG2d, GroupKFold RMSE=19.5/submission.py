"""
submission.py
近赤外スペクトルから含水率を予測する提出ファイル生成スクリプト

モデル: CatBoost(Optuna) × 0.70 + LightGBM × 0.30 アンサンブル
前処理: SG2d_only (window=21, polyorder=3, deriv=2)
CV: GroupKFold RMSE ≈ 19.5
"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from datetime import datetime
import os
import json

# ============================================================
# 設定
# ============================================================
RANDOM_SEED = 42
N_SPLITS = 5

# Optunaで見つけたベストパラメータ
CATBOOST_PARAMS = {
    "iterations": 2000,
    "learning_rate": 0.012064,
    "depth": 3,
    "l2_leaf_reg": 8.441087,
    "bagging_temperature": 0.599068,
    "random_strength": 1.279867,
    "border_count": 47,
    "random_seed": RANDOM_SEED,
    "verbose": 0,
    "early_stopping_rounds": 100,
    "task_type": "GPU",  # GPUがなければ "CPU" に変更
}

LGBM_PARAMS = {
    "objective": "regression",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "random_state": RANDOM_SEED,
}

ENSEMBLE_WEIGHT_CAT = 0.70
ENSEMBLE_WEIGHT_LGB = 0.30

# ============================================================
# 前処理
# ============================================================
def sg_deriv(X, window=21, polyorder=3, deriv=2):
    return savgol_filter(X, window_length=window, polyorder=polyorder,
                         deriv=deriv, axis=1)

# ============================================================
# メイン処理
# ============================================================
def main():
    print("=" * 60)
    print("📂 データ読み込み")
    print("=" * 60)

    # --- 学習データ ---
    train_df = pd.read_csv("../../data/train.csv", encoding="cp932")
    
    target_col = "含水率"
    meta_cols = ["sample number", "species number", "樹種", "含水率"]
    spectrum_cols = [c for c in train_df.columns if c not in meta_cols]

    X_train_raw = train_df[spectrum_cols].to_numpy(dtype=float)
    y_train = train_df[target_col].values
    species_train = train_df["species number"].values

    print(f"  学習データ: {X_train_raw.shape}")

    # --- テストデータ ---
    test_df = pd.read_csv("../../data/test.csv", encoding="cp932")
    
    # テストデータにも同じメタ列があるはずだが、含水率はない
    test_meta_cols = ["sample number", "species number", "樹種"]
    test_spectrum_cols = [c for c in test_df.columns if c not in test_meta_cols]
    
    X_test_raw = test_df[test_spectrum_cols].to_numpy(dtype=float)
    
    print(f"  テストデータ: {X_test_raw.shape}")
    print(f"  テスト樹種: {test_df['樹種'].unique()}")

    # --- 前処理 (SG2d_only) ---
    print("\n🔧 前処理: SG2d (window=21, polyorder=3)")
    X_train = sg_deriv(X_train_raw)
    X_test = sg_deriv(X_test_raw)

    # ============================================================
    # 学習（全訓練データで複数モデルを学習 → 平均）
    # ============================================================
    print("\n" + "=" * 60)
    print("🚀 モデル学習")
    print("=" * 60)

    # --- 方法: GroupKFold で学習した5つのモデルの予測を平均 ---
    # (全データで1モデルより、CV学習済みモデルの平均の方がロバスト)
    
    cv = GroupKFold(n_splits=N_SPLITS)
    
    cat_predictions = []
    lgb_predictions = []
    
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train, groups=species_train)):
        print(f"\n  --- Fold {fold} ---")
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        
        # CatBoost
        cat_model = CatBoostRegressor(**CATBOOST_PARAMS)
        cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
        cat_pred = cat_model.predict(X_test)
        cat_predictions.append(cat_pred)
        
        cat_va_rmse = np.sqrt(np.mean((cat_model.predict(X_va) - y_va)**2))
        print(f"    CatBoost val RMSE: {cat_va_rmse:.2f}")
        
        # LightGBM
        lgb_model = LGBMRegressor(**LGBM_PARAMS)
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[early_stopping(50), log_evaluation(0)]
        )
        lgb_pred = lgb_model.predict(X_test)
        lgb_predictions.append(lgb_pred)
        
        lgb_va_rmse = np.sqrt(np.mean((lgb_model.predict(X_va) - y_va)**2))
        print(f"    LightGBM val RMSE: {lgb_va_rmse:.2f}")

    # --- 5 foldの予測を平均 ---
    cat_mean = np.mean(cat_predictions, axis=0)
    lgb_mean = np.mean(lgb_predictions, axis=0)
    
    # --- アンサンブル ---
    final_pred = ENSEMBLE_WEIGHT_CAT * cat_mean + ENSEMBLE_WEIGHT_LGB * lgb_mean

    # 含水率は0以上のはず
    final_pred = np.clip(final_pred, 0, None)

    # ============================================================
    # 提出ファイル生成
    # ============================================================
    print("\n" + "=" * 60)
    print("📝 提出ファイル生成")
    print("=" * 60)

    # ファイル名: 日付_モデル名
    today = datetime.now().strftime("%Y%m%d")
    model_desc = "cat070_lgb030_SG2d_optuna"
    filename = f"sub_{today}_{model_desc}.csv"

    submission = pd.DataFrame({
        "sample number": test_df["sample number"],
        "含水率": final_pred,
    })

    submission.to_csv(filename, index=False, header=False, encoding="cp932")
    
    print(f"  ファイル名:     {filename}")
    print(f"  テストサンプル数: {len(submission)}")
    print(f"  予測含水率範囲:  {final_pred.min():.1f} ~ {final_pred.max():.1f} %")
    print(f"  予測含水率平均:  {final_pred.mean():.1f} %")

    # --- 予測の概要を表示 ---
    print("\n  樹種別の予測含水率平均:")
    submission_with_species = submission.copy()
    submission_with_species["樹種"] = test_df["樹種"].values
    species_summary = submission_with_species.groupby("樹種")["含水率"].agg(["mean", "min", "max", "count"])
    print(species_summary.round(1).to_string())

    # --- メタ情報をjsonで保存（再現性のため）---
    meta = {
        "date": today,
        "model": model_desc,
        "preprocess": "SG2d_only(window=21, polyorder=3)",
        "catboost_params": {k: v for k, v in CATBOOST_PARAMS.items() if k != "task_type"},
        "lgbm_params": LGBM_PARAMS,
        "ensemble_weight": {"catboost": ENSEMBLE_WEIGHT_CAT, "lightgbm": ENSEMBLE_WEIGHT_LGB},
        "cv_rmse_approx": 19.5,
        "n_folds": N_SPLITS,
    }
    meta_filename = f"sub_{today}_{model_desc}_meta.json"
    with open(meta_filename, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n  メタ情報保存: {meta_filename}")

    print("\n✅ 完了！")
    print(f"  提出ファイル: {filename}")

if __name__ == "__main__":
    main()