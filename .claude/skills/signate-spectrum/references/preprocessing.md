# 前処理

## スペクトル列の取得

```python
meta_cols_train = ['sample number', 'species number', '樹種', '含水率']
meta_cols_test  = ['sample number', 'species number', '樹種']
spectrum_cols = [c for c in train.columns if c not in meta_cols_train]

X_train = train[spectrum_cols].values
y_train = train['含水率'].values
X_test  = test[spectrum_cols].values
```

## 主要な前処理手法

### 1. SNV（Standard Normal Variate）
散乱補正。各サンプルを平均0・標準偏差1に標準化。

```python
def snv(X):
    return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

X_snv = snv(X_train)
```

### 2. MSC（Multiplicative Scatter Correction）
リファレンス（平均スペクトル）に対するスケール・オフセット補正。

```python
import numpy as np

def msc(X, reference=None):
    if reference is None:
        reference = X.mean(axis=0)
    X_msc = np.zeros_like(X)
    for i in range(X.shape[0]):
        coef = np.polyfit(reference, X[i], 1)
        X_msc[i] = (X[i] - coef[1]) / coef[0]
    return X_msc, reference

X_msc, ref = msc(X_train)
X_test_msc, _ = msc(X_test, reference=ref)  # trainのrefを使う
```

### 3. Savitzky-Golay 微分
ノイズ除去しながら1次・2次微分を取る。

```python
from scipy.signal import savgol_filter

# 1次微分
X_sg1 = savgol_filter(X_train, window_length=11, polyorder=2, deriv=1)
# 2次微分
X_sg2 = savgol_filter(X_train, window_length=11, polyorder=2, deriv=2)
```

### 4. 標準化（StandardScaler）

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # trainのscalerを使う
```

### 5. 特定波数域のトリミング
水分吸収帯（約5200 cm⁻¹, 6900 cm⁻¹付近）に絞る場合。

```python
import numpy as np
wavenumbers = np.array([float(c) for c in spectrum_cols])

# 例: 5000〜7500 cm⁻¹に絞る
mask = (wavenumbers >= 5000) & (wavenumbers <= 7500)
X_trimmed = X_train[:, mask]
```

## 注意点

- testデータの前処理は必ずtrainで fit したパラメータを使うこと
- MSC の reference は train の平均スペクトルを保存・再利用する
- 微分を適用するとエッジ効果が生じるため window_length に注意
