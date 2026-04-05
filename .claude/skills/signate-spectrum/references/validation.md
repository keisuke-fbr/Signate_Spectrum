# バリデーション戦略

## 重要な前提

**trainとtestで樹種が完全に異なる。**
通常のランダムCV（KFold）では同一樹種内でtrain/valが分割されるため、
モデルが樹種固有のパターンを学習してしまい、未知樹種への汎化性能を正しく評価できない。

## 推奨：GroupKFold（樹種単位）

```python
from sklearn.model_selection import GroupKFold
import numpy as np

groups = train['樹種'].values
gkf = GroupKFold(n_splits=len(train['樹種'].unique()))  # 13-fold（樹種数）

oof_preds = np.zeros(len(train))

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model.fit(X_tr, y_tr)
    oof_preds[val_idx] = model.predict(X_val)

    fold_rmse = np.sqrt(np.mean((y_val - oof_preds[val_idx]) ** 2))
    val_species = train['樹種'].iloc[val_idx].iloc[0]
    print(f"Fold {fold+1} ({val_species}): RMSE = {fold_rmse:.4f}")

overall_rmse = np.sqrt(np.mean((y_train - oof_preds) ** 2))
print(f"\nOOF RMSE: {overall_rmse:.4f}")
```

## RMSE 計算

```python
import numpy as np
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
```

## バリデーション設計の考え方

| 方法 | 用途 | 注意 |
|------|------|------|
| GroupKFold（樹種） | 汎化性能の評価 | 推奨。未知樹種への性能を反映 |
| KFold（ランダム） | 使用不可 | 同一樹種内分割でリークが起きる |
| Hold-out（特定樹種） | 最終確認 | 特定樹種1つをvalidに固定 |

## 注意点

- OOFスコアがLBスコアと乖離する場合は前処理か特徴量に問題あり
- 樹種ごとのfold RMSEを確認し、特定樹種で極端に悪いfoldがないか確認する
