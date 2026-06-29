import json, uuid

with open("notebook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

def make_cell(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }

# ── 既存 Cell 0〜3 を保持 ────────────────────────────────────────────────────
cells = nb["cells"][:4]

# ── Cell 0 を imports 追加で更新 ──────────────────────────────────────────────
cells[0] = make_cell("""\
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import warnings
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
""")

# ── Cell 4: 特徴量ディスパッチ + LightGBM ────────────────────────────────────
cells.append(make_cell("""\
# ============================================================
# 3. 特徴量ディスパッチ + LightGBM ランナー
# ============================================================

def _get_features(feat_set, X_tr, y_tr, X_aug, X_va, X_te, n_orig):
    \"\"\"特徴量セット A/B/C へのディスパッチ関数。\"\"\"
    if feat_set == 'A':
        return get_feature_A(X_tr, y_tr, X_aug, X_va, X_te, n_orig)
    elif feat_set == 'B':
        return get_feature_B(X_aug, X_va, X_te)
    elif feat_set == 'C':
        return get_feature_C(X_tr, X_aug, X_va, X_te)
    else:
        raise ValueError(f"Unknown feat_set: {feat_set}")


DEFAULT_LGB_PARAMS = {
    'n_estimators'    : 1500,
    'learning_rate'   : 0.02,
    'max_depth'       : 4,
    'num_leaves'      : 15,
    'subsample'       : 0.8,
    'colsample_bytree': 0.6,
    'min_child_samples': 20,
    'reg_alpha'       : 0.2,
    'reg_lambda'      : 3.0,
}


def run_lgb(feat_set, params=None, seed=42):
    \"\"\"LightGBM ランナー（特徴量セット A/B/C に対応）。

    リーク防止:
        - Mixup は GroupKFold 分割後の訓練フォールド (X_tr) のみで生成
        - 特徴量の fit（PLS・PCA・KNN）はすべて X_tr のみで実施
        - val・test には transform のみ適用
    \"\"\"
    if params is None:
        params = DEFAULT_LGB_PARAMS

    gkf = GroupKFold(n_splits=5)
    oof_pred   = np.zeros(len(train))
    final_pred = np.zeros(len(test))

    for fold, (tr_idx, va_idx) in enumerate(
            gkf.split(X_train_raw, y_train_log, groups)):

        X_tr  = X_train_raw[tr_idx]
        y_tr  = y_train_log.iloc[tr_idx].values
        X_va  = X_train_raw[va_idx]
        y_va  = y_train_log.iloc[va_idx].values
        tr_sp = groups.iloc[tr_idx].values

        # Mixup: 訓練フォールドのみ使用（リーク防止）
        X_mix, y_mix = mixup_hybrid(
            X_tr, y_tr, tr_sp, seed=seed + fold)
        n_orig = len(X_tr)
        X_aug  = np.vstack([X_tr, X_mix])
        y_aug  = np.concatenate([y_tr, y_mix])

        # 特徴量取得（内部で fit は X_tr のみ）
        F_aug, F_va, F_te = _get_features(
            feat_set, X_tr, y_tr, X_aug, X_va, X_test_raw, n_orig)

        model = lgb.LGBMRegressor(
            **params, random_state=seed, verbosity=-1)
        model.fit(F_aug, y_aug,
                  eval_set=[(F_va, y_va)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

        oof_pred[va_idx] = np.expm1(model.predict(F_va))
        final_pred      += np.expm1(model.predict(F_te)) / 5

    oof_rmse = np.sqrt(mean_squared_error(y_true, oof_pred))
    return {'oof_pred': oof_pred, 'test_pred': final_pred, 'oof_rmse': oof_rmse}
"""))

# ── Cell 5: SVR ───────────────────────────────────────────────────────────────
cells.append(make_cell("""\
# ============================================================
# 4. SVR ランナー（RBF カーネル）
# ============================================================

DEFAULT_SVR_PARAMS = {
    'kernel' : 'rbf',
    'C'      : 10.0,
    'epsilon': 0.1,
    'gamma'  : 'scale',
}


def run_svr(feat_set, params=None, seed=42):
    \"\"\"SVR ランナー（特徴量セット A/B/C に対応）。

    リーク防止:
        - Mixup・特徴量 fit は run_lgb と同じ設計
        - StandardScaler は X_tr 由来の特徴量（F_aug[:n_orig]）のみで fit
          → val・test には transform のみ適用
    \"\"\"
    if params is None:
        params = DEFAULT_SVR_PARAMS

    gkf = GroupKFold(n_splits=5)
    oof_pred   = np.zeros(len(train))
    final_pred = np.zeros(len(test))

    for fold, (tr_idx, va_idx) in enumerate(
            gkf.split(X_train_raw, y_train_log, groups)):

        X_tr  = X_train_raw[tr_idx]
        y_tr  = y_train_log.iloc[tr_idx].values
        X_va  = X_train_raw[va_idx]
        y_va  = y_train_log.iloc[va_idx].values
        tr_sp = groups.iloc[tr_idx].values

        # Mixup: 訓練フォールドのみ（リーク防止）
        X_mix, y_mix = mixup_hybrid(
            X_tr, y_tr, tr_sp, seed=seed + fold)
        n_orig = len(X_tr)
        X_aug  = np.vstack([X_tr, X_mix])
        y_aug  = np.concatenate([y_tr, y_mix])

        # 特徴量取得（内部 fit は X_tr のみ）
        F_aug, F_va, F_te = _get_features(
            feat_set, X_tr, y_tr, X_aug, X_va, X_test_raw, n_orig)

        # スケーリング: X_tr 由来の特徴量のみで fit（リーク防止）
        scaler = StandardScaler()
        scaler.fit(F_aug[:n_orig])
        F_aug_s = scaler.transform(F_aug)
        F_va_s  = scaler.transform(F_va)
        F_te_s  = scaler.transform(F_te)

        model = SVR(**params)
        model.fit(F_aug_s, y_aug)

        oof_pred[va_idx] = np.expm1(model.predict(F_va_s))
        final_pred      += np.expm1(model.predict(F_te_s)) / 5

    oof_rmse = np.sqrt(mean_squared_error(y_true, oof_pred))
    return {'oof_pred': oof_pred, 'test_pred': final_pred, 'oof_rmse': oof_rmse}
"""))

# ── Cell 6: Ridge ─────────────────────────────────────────────────────────────
cells.append(make_cell("""\
# ============================================================
# 5. Ridge Regression ランナー
# ============================================================

DEFAULT_RIDGE_PARAMS = {
    'alpha': 1.0,
}


def run_ridge_model(feat_set, params=None):
    \"\"\"Ridge Regression ランナー（特徴量セット A/B/C に対応）。

    リーク防止:
        - Mixup・特徴量 fit は run_lgb と同じ設計
        - StandardScaler は F_aug[:n_orig] のみで fit
    \"\"\"
    if params is None:
        params = DEFAULT_RIDGE_PARAMS

    gkf = GroupKFold(n_splits=5)
    oof_pred   = np.zeros(len(train))
    final_pred = np.zeros(len(test))

    for fold, (tr_idx, va_idx) in enumerate(
            gkf.split(X_train_raw, y_train_log, groups)):

        X_tr  = X_train_raw[tr_idx]
        y_tr  = y_train_log.iloc[tr_idx].values
        X_va  = X_train_raw[va_idx]
        y_va  = y_train_log.iloc[va_idx].values
        tr_sp = groups.iloc[tr_idx].values

        # Mixup: 訓練フォールドのみ（リーク防止）
        X_mix, y_mix = mixup_hybrid(X_tr, y_tr, tr_sp, seed=42 + fold)
        n_orig = len(X_tr)
        X_aug  = np.vstack([X_tr, X_mix])
        y_aug  = np.concatenate([y_tr, y_mix])

        # 特徴量取得（内部 fit は X_tr のみ）
        F_aug, F_va, F_te = _get_features(
            feat_set, X_tr, y_tr, X_aug, X_va, X_test_raw, n_orig)

        # スケーリング: X_tr 由来の特徴量のみで fit（リーク防止）
        scaler = StandardScaler()
        scaler.fit(F_aug[:n_orig])
        F_aug_s = scaler.transform(F_aug)
        F_va_s  = scaler.transform(F_va)
        F_te_s  = scaler.transform(F_te)

        model = Ridge(**params)
        model.fit(F_aug_s, y_aug)

        oof_pred[va_idx] = np.expm1(model.predict(F_va_s))
        final_pred      += np.expm1(model.predict(F_te_s)) / 5

    oof_rmse = np.sqrt(mean_squared_error(y_true, oof_pred))
    return {'oof_pred': oof_pred, 'test_pred': final_pred, 'oof_rmse': oof_rmse}
"""))

# ── Cell 7: Optuna チューニング ───────────────────────────────────────────────
cells.append(make_cell("""\
# ============================================================
# 6. Optuna ハイパーパラメータチューニング
# ============================================================
# リーク防止の設計:
#   - objective 関数内で GroupKFold を毎回実行
#   - val は特定のフォールドに固定せず OOF RMSE を最小化
#   - test データは objective 関数内で一切使用しない

def _oof_rmse(oof_log):
    \"\"\"log スケールの OOF 予測から RMSE を計算。\"\"\"
    return np.sqrt(mean_squared_error(y_true, np.expm1(oof_log)))


# ── LightGBM objective ──────────────────────────────────────
def objective_lgb(trial):
    params = {
        'n_estimators'    : 1500,
        'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth'       : trial.suggest_int('max_depth', 3, 6),
        'num_leaves'      : trial.suggest_int('num_leaves', 10, 50),
        'subsample'       : trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'reg_alpha'       : trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
        'reg_lambda'      : trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
    }

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(train))

    for fold, (tr_idx, va_idx) in enumerate(
            gkf.split(X_train_raw, y_train_log, groups)):
        X_tr  = X_train_raw[tr_idx]
        y_tr  = y_train_log.iloc[tr_idx].values
        X_va  = X_train_raw[va_idx]
        y_va  = y_train_log.iloc[va_idx].values
        tr_sp = groups.iloc[tr_idx].values

        X_mix, y_mix = mixup_hybrid(X_tr, y_tr, tr_sp, seed=42 + fold)
        n_orig = len(X_tr)
        X_aug  = np.vstack([X_tr, X_mix])
        y_aug  = np.concatenate([y_tr, y_mix])

        # 特徴量A で評価（代表）
        F_aug, F_va, _ = get_feature_A(
            X_tr, y_tr, X_aug, X_va, X_test_raw, n_orig)

        model = lgb.LGBMRegressor(**params, random_state=42, verbosity=-1)
        model.fit(F_aug, y_aug,
                  eval_set=[(F_va, y_va)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va_idx] = model.predict(F_va)

    return _oof_rmse(oof)


# ── SVR objective ────────────────────────────────────────────
def objective_svr(trial):
    params = {
        'kernel' : 'rbf',
        'C'      : trial.suggest_float('C', 0.1, 100.0, log=True),
        'epsilon': trial.suggest_float('epsilon', 0.01, 1.0, log=True),
        'gamma'  : trial.suggest_categorical('gamma', ['scale', 'auto']),
    }

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(train))

    for fold, (tr_idx, va_idx) in enumerate(
            gkf.split(X_train_raw, y_train_log, groups)):
        X_tr  = X_train_raw[tr_idx]
        y_tr  = y_train_log.iloc[tr_idx].values
        X_va  = X_train_raw[va_idx]
        tr_sp = groups.iloc[tr_idx].values

        X_mix, y_mix = mixup_hybrid(X_tr, y_tr, tr_sp, seed=42 + fold)
        n_orig = len(X_tr)
        X_aug  = np.vstack([X_tr, X_mix])
        y_aug  = np.concatenate([y_tr, y_mix])

        F_aug, F_va, _ = get_feature_A(
            X_tr, y_tr, X_aug, X_va, X_test_raw, n_orig)

        scaler = StandardScaler()
        scaler.fit(F_aug[:n_orig])
        F_aug_s = scaler.transform(F_aug)
        F_va_s  = scaler.transform(F_va)

        model = SVR(**params)
        model.fit(F_aug_s, y_aug)
        oof[va_idx] = model.predict(F_va_s)

    return _oof_rmse(oof)


# ── Ridge objective ──────────────────────────────────────────
def objective_ridge(trial):
    params = {
        'alpha': trial.suggest_float('alpha', 1e-3, 100.0, log=True),
    }

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(train))

    for fold, (tr_idx, va_idx) in enumerate(
            gkf.split(X_train_raw, y_train_log, groups)):
        X_tr  = X_train_raw[tr_idx]
        y_tr  = y_train_log.iloc[tr_idx].values
        X_va  = X_train_raw[va_idx]
        tr_sp = groups.iloc[tr_idx].values

        X_mix, y_mix = mixup_hybrid(X_tr, y_tr, tr_sp, seed=42 + fold)
        n_orig = len(X_tr)
        X_aug  = np.vstack([X_tr, X_mix])
        y_aug  = np.concatenate([y_tr, y_mix])

        # Ridge は高次元向きなので特徴量C で評価
        F_aug, F_va, _ = get_feature_C(X_tr, X_aug, X_va, X_test_raw)

        scaler = StandardScaler()
        scaler.fit(F_aug[:n_orig])
        F_aug_s = scaler.transform(F_aug)
        F_va_s  = scaler.transform(F_va)

        model = Ridge(**params)
        model.fit(F_aug_s, y_aug)
        oof[va_idx] = model.predict(F_va_s)

    return _oof_rmse(oof)


# ── Optuna 実行 ──────────────────────────────────────────────
print("Optuna チューニング開始...")

study_lgb = optuna.create_study(direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
study_lgb.optimize(objective_lgb, n_trials=50, show_progress_bar=True)
best_params_lgb = {**DEFAULT_LGB_PARAMS, **study_lgb.best_params}
print(f"LGB  best OOF: {study_lgb.best_value:.4f}")
print(f"     best params: {study_lgb.best_params}")

study_svr = optuna.create_study(direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
study_svr.optimize(objective_svr, n_trials=30, show_progress_bar=True)
best_params_svr = {**DEFAULT_SVR_PARAMS, **study_svr.best_params}
print(f"SVR  best OOF: {study_svr.best_value:.4f}")
print(f"     best params: {study_svr.best_params}")

study_ridge = optuna.create_study(direction='minimize',
                                   sampler=optuna.samplers.TPESampler(seed=42))
study_ridge.optimize(objective_ridge, n_trials=20, show_progress_bar=True)
best_params_ridge = {**DEFAULT_RIDGE_PARAMS, **study_ridge.best_params}
print(f"Ridge best OOF: {study_ridge.best_value:.4f}")
print(f"      best params: {study_ridge.best_params}")
"""))

# ── Cell 8: 全9モデル実行 ─────────────────────────────────────────────────────
cells.append(make_cell("""\
# ============================================================
# 7. 全9モデルの実行
# ============================================================
# アルゴリズム × 特徴量セット の 3×3 = 9 モデルを実行
# 各モデルは OOF 予測とテスト予測を返す

print("=" * 60)
print("全9モデルを実行します")
print("=" * 60)

results = {}

# ── LightGBM × 3 ────────────────────────────────────────────
print("\\n[LightGBM]")
for feat in ['A', 'B', 'C']:
    key = f'lgb_{feat}'
    print(f"  {key} ...", end=" ", flush=True)
    results[key] = run_lgb(feat, params=best_params_lgb, seed=42)
    print(f"OOF RMSE = {results[key]['oof_rmse']:.4f}")

# ── SVR × 3 ─────────────────────────────────────────────────
print("\\n[SVR]")
for feat in ['A', 'B', 'C']:
    key = f'svr_{feat}'
    print(f"  {key} ...", end=" ", flush=True)
    results[key] = run_svr(feat, params=best_params_svr, seed=42)
    print(f"OOF RMSE = {results[key]['oof_rmse']:.4f}")

# ── Ridge × 3 ───────────────────────────────────────────────
print("\\n[Ridge]")
for feat in ['A', 'B', 'C']:
    key = f'ridge_{feat}'
    print(f"  {key} ...", end=" ", flush=True)
    results[key] = run_ridge_model(feat, params=best_params_ridge)
    print(f"OOF RMSE = {results[key]['oof_rmse']:.4f}")

# ── 結果一覧 ────────────────────────────────────────────────
print("\\n" + "=" * 60)
print("OOF RMSE 一覧")
print("=" * 60)
for k in sorted(results, key=lambda x: results[x]['oof_rmse']):
    print(f"  {k:<12s}  {results[k]['oof_rmse']:.4f}")
"""))

# ── Cell 9: スタッキング ──────────────────────────────────────────────────────
cells.append(make_cell("""\
# ============================================================
# 8. スタッキング（Ridge メタ学習器）
# ============================================================
# Level 0: 9モデルの OOF 予測（各サンプルに1回ずつの予測）
# Level 1: Ridge がその9列を入力して最終予測を出す
#
# リーク防止:
#   OOF 予測はすでにリークフリー（各サンプルを見ていない
#   モデルが予測した値）なので、そのまま Ridge に投入して良い

model_keys = [f'{algo}_{feat}'
              for algo in ['lgb', 'svr', 'ridge']
              for feat in ['A', 'B', 'C']]

# Level 0 の出力を行列に変換
oof_matrix  = np.column_stack([results[k]['oof_pred']  for k in model_keys])
test_matrix = np.column_stack([results[k]['test_pred'] for k in model_keys])
# shape: (n_train, 9), (n_test, 9)

# メタ特徴量のスケーリング（Ridge は距離に敏感）
meta_scaler = StandardScaler()
oof_s  = meta_scaler.fit_transform(oof_matrix)
test_s = meta_scaler.transform(test_matrix)

# Ridge メタ学習器を OOF で学習
meta_model = Ridge(alpha=1.0)
meta_model.fit(oof_s, y_true)

# テスト予測
stacking_pred = meta_model.predict(test_s)
stacking_pred = np.clip(stacking_pred, 0, None)

# OOF での擬似スコア確認（参考値）
oof_stacking = meta_model.predict(oof_s)
oof_stacking = np.clip(oof_stacking, 0, None)
oof_rmse_stack = np.sqrt(mean_squared_error(y_true, oof_stacking))

print("=" * 60)
print("スタッキング結果")
print("=" * 60)
print(f"  メタ学習器: Ridge (alpha=1.0)")
print(f"  入力モデル数: {len(model_keys)}")
print(f"  OOF RMSE（参考）: {oof_rmse_stack:.4f}")
print()
print("  各モデルの重み（係数）:")
for k, coef in zip(model_keys, meta_model.coef_):
    print(f"    {k:<12s}  {coef:+.4f}")
"""))

# ── Cell 10: 提出ファイル出力 ─────────────────────────────────────────────────
cells.append(make_cell("""\
# ============================================================
# 9. 提出ファイルの出力
# ============================================================
output_df = submit_template.copy()
output_df[1] = stacking_pred

output_filename = "koyama_stacking_submission.csv"
output_df.to_csv(output_filename, index=False, header=False)

print(f"出力ファイル: {output_filename}")
print(f"予測値の範囲: {stacking_pred.min():.2f} 〜 {stacking_pred.max():.2f}")
print(f"予測値の平均: {stacking_pred.mean():.2f}")
"""))

# ── ノートブックを保存 ────────────────────────────────────────────────────────
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python (signate-spectrum)",
        "language": "python",
        "name": "signate-spectrum"
    },
    "language_info": {"name": "python", "version": "3.9.25"}
}
with open("notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"notebook.ipynb 更新完了: {len(nb['cells'])} cells")
print()
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    funcs = [l.strip() for l in src.split("\n") if l.startswith("def ")]
    first = src.split("\n")[0][:55]
    print(f"  Cell {i:2d}: {first}")
    for f in funcs:
        print(f"           {f}")
