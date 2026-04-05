# モデリング

## モデル候補一覧

| アプローチ | 手法 | 特徴 |
|-----------|------|------|
| ケモメトリクス | PLS | 解釈性高・次元削減・少サンプルに強い |
| 線形 | Ridge / Lasso | シンプル・高速・正則化で過学習抑制 |
| カーネル | SVR | 非線形対応・特徴量スケーリング必須 |
| アンサンブル | Random Forest | 特徴量重要度取得可 |
| 勾配ブースティング | LightGBM | 高精度・高速・カテゴリ変数対応 |
| NN | 1D-CNN / MLP | 大データ向け・チューニングコスト高 |

## PLS（Partial Least Squares）

```python
from sklearn.cross_decomposition import PLSRegression

pls = PLSRegression(n_components=10)
pls.fit(X_train, y_train)
y_pred = pls.predict(X_test).flatten()
```

## LightGBM

```python
import lightgbm as lgb

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbose': -1,
}

model = lgb.LGBMRegressor(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)
y_pred = model.predict(X_test)
```

## Ridge

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
```

## 提出ファイル生成

```python
import pandas as pd

submission = pd.DataFrame({
    'sample number': test['sample number'],
    '含水率': y_pred
})
submission.to_csv('submission.csv', index=False, encoding='utf-8-sig')
```
