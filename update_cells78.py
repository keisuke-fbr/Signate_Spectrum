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

# Cell 0-6 は保持、Cell 7-10 を更新
cells = nb["cells"][:7]

# ── Cell 7: Optuna (9モデル独立チューニング) ───────────────────────────────
cells.append(make_cell("""\
# ============================================================
# 6. Optuna ハイパーパラメータチューニング（9モデル独立）
# ============================================================
# 各 (アルゴリズム × 特徴量) の組み合わせごとに独立して最適化
# リーク防止: objective 内で GroupKFold を毎回実行、test は不使用

def _oof_rmse(oof_log):
    return np.sqrt(mean_squared_error(y_true, np.expm1(oof_log)))


# ── LightGBM objective factory ───────────────────────────────
def make_objective_lgb(feat_set):
    def objective(trial):
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
            F_aug, F_va, _ = _get_features(
                feat_set, X_tr, y_tr, X_aug, X_va, X_test_raw, n_orig)
            model = lgb.LGBMRegressor(**params, random_state=42, verbosity=-1)
            model.fit(F_aug, y_aug,
                      eval_set=[(F_va, y_va)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
            oof[va_idx] = model.predict(F_va)
        return _oof_rmse(oof)
    return objective


# ── SVR objective factory ────────────────────────────────────
def make_objective_svr(feat_set):
    def objective(trial):
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
            F_aug, F_va, _ = _get_features(
                feat_set, X_tr, y_tr, X_aug, X_va, X_test_raw, n_orig)
            scaler = StandardScaler()
            scaler.fit(F_aug[:n_orig])
            model = SVR(**params)
            model.fit(scaler.transform(F_aug), y_aug)
            oof[va_idx] = model.predict(scaler.transform(F_va))
        return _oof_rmse(oof)
    return objective


# ── Ridge objective factory ──────────────────────────────────
def make_objective_ridge(feat_set):
    def objective(trial):
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
            F_aug, F_va, _ = _get_features(
                feat_set, X_tr, y_tr, X_aug, X_va, X_test_raw, n_orig)
            scaler = StandardScaler()
            scaler.fit(F_aug[:n_orig])
            model = Ridge(**params)
            model.fit(scaler.transform(F_aug), y_aug)
            oof[va_idx] = model.predict(scaler.transform(F_va))
        return _oof_rmse(oof)
    return objective


# ── 9モデル独立チューニング ──────────────────────────────────
MODEL_CONFIGS = [
    ('lgb',   'A', make_objective_lgb,   50),
    ('lgb',   'B', make_objective_lgb,   50),
    ('lgb',   'C', make_objective_lgb,   50),
    ('svr',   'A', make_objective_svr,   30),
    ('svr',   'B', make_objective_svr,   30),
    ('svr',   'C', make_objective_svr,   30),
    ('ridge', 'A', make_objective_ridge, 20),
    ('ridge', 'B', make_objective_ridge, 20),
    ('ridge', 'C', make_objective_ridge, 20),
]

DEFAULT_PARAMS = {
    'lgb'  : DEFAULT_LGB_PARAMS,
    'svr'  : DEFAULT_SVR_PARAMS,
    'ridge': DEFAULT_RIDGE_PARAMS,
}

best_params = {}

print("Optuna チューニング開始（9モデル独立）...")
print("=" * 60)

for algo, feat, make_fn, n_trials in MODEL_CONFIGS:
    key = f'{algo}_{feat}'
    print(f"\\n[{key}] {n_trials} trials ...")
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(make_fn(feat), n_trials=n_trials, show_progress_bar=True)
    best_params[key] = {**DEFAULT_PARAMS[algo], **study.best_params}
    print(f"  best OOF RMSE : {study.best_value:.4f}")
    print(f"  best params   : {study.best_params}")

print("\\n" + "=" * 60)
print("チューニング完了")
"""))

# ── Cell 8: 全9モデル実行（per-model params 使用） ────────────────────────────
cells.append(make_cell("""\
# ============================================================
# 7. 全9モデルの実行（モデルごとの最適パラメータを使用）
# ============================================================
print("=" * 60)
print("全9モデルを実行します")
print("=" * 60)

results = {}

print("\\n[LightGBM]")
for feat in ['A', 'B', 'C']:
    key = f'lgb_{feat}'
    print(f"  {key} ...", end=" ", flush=True)
    results[key] = run_lgb(feat, params=best_params[key], seed=42)
    print(f"OOF RMSE = {results[key]['oof_rmse']:.4f}")

print("\\n[SVR]")
for feat in ['A', 'B', 'C']:
    key = f'svr_{feat}'
    print(f"  {key} ...", end=" ", flush=True)
    results[key] = run_svr(feat, params=best_params[key], seed=42)
    print(f"OOF RMSE = {results[key]['oof_rmse']:.4f}")

print("\\n[Ridge]")
for feat in ['A', 'B', 'C']:
    key = f'ridge_{feat}'
    print(f"  {key} ...", end=" ", flush=True)
    results[key] = run_ridge_model(feat, params=best_params[key])
    print(f"OOF RMSE = {results[key]['oof_rmse']:.4f}")

print("\\n" + "=" * 60)
print("OOF RMSE 一覧")
print("=" * 60)
for k in sorted(results, key=lambda x: results[x]['oof_rmse']):
    print(f"  {k:<12s}  {results[k]['oof_rmse']:.4f}")
"""))

# ── Cell 9: スタッキング (変更なし) ──────────────────────────────────────────
cells.append(nb["cells"][9])

# ── Cell 10: 出力 (変更なし) ─────────────────────────────────────────────────
cells.append(nb["cells"][10])

nb["cells"] = cells
with open("notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"更新完了: {len(nb['cells'])} cells")
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    funcs = [l.strip() for l in src.split("\n") if l.startswith("def ")]
    first = src.split("\n")[0][:55]
    print(f"  Cell {i:2d}: {first}")
    for fn in funcs:
        print(f"           {fn}")
