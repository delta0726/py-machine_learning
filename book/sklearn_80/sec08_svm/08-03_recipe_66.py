# ******************************************************************************
# Chapter   : 8 サポートベクトルマシン
# Title     : 8-3 SVMを最適化する（Recipe66)
# Created by: Owner
# Created on: 2020/12/30
# Page      : P246 - P252
# ******************************************************************************

# ＜概要＞
#


# ＜目次＞
# 0 準備
# 1 パイプラインの構築
# 2 グリッドサーチ・チューニング


# 0 準備 ------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from itertools import product


# データロード
iris = datasets.load_iris()

# データ格納
X_w = iris.data[:, :2]
y_w = iris.target

X = X_w[y_w != 0]
y = y_w[y_w != 0]

X_1 = X[y == 1]
X_2 = X[y == 2]

# データ分割
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)


# 1 パイプラインの構築 ----------------------------------------------------------------------

# インスタンス生成
svm_est = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())
])

# パラメータ設定
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1, 10]
param_grid = dict(svc__estimator__gamma=gammas, svc__estimator__C=Cs)


# 2 グリッドサーチ・チューニング -------------------------------------------------------------

# インスタンス生成
# --- クロスバリデーション
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=7)

# インスタンス生成
# --- グリッドサーチ
grid_cv = GridSearchCV(svm_est, param_grid=param_grid, cv=cv)

# チューニング実行
grid_cv.fit(X_train, y_train)

# 最良パラメータ
grid_cv.best_params_

# 最良スコア
grid_cv.best_score_


# 3 ランダムサーチ・チューニング -------------------------------------------------------------

# インスタンス生成
# --- ランダムサーチ
rand_grid = RandomizedSearchCV(svm_est, param_distributions=param_grid, cv=cv, n_iter=10)

# チューニング実行
rand_grid.fit(X_train, y_train)

# 最良パラメータ
rand_grid.best_params_


# 4 モデル可視化 -------------------------------------------------------------------------

# Minima and maxima of both features
xmin, xmax = np.percentile(X[:, 0], [0, 100])
ymin, ymax = np.percentile(X[:, 1], [0, 100])

# Grid/Cartesian product with itertools.product
test_points = np.array([[xx, yy] for xx, yy in product(np.linspace(xmin, xmax), np.linspace(ymin, ymax))])

# Predictions on the grid
test_preds = grid_cv.predict(test_points)


X_1 = X[y == 1]
X_2 = X[y == 2]

plt.figure(figsize=(10,7))
plt.scatter(X_2[:, 0], X_2[:, 1], color='red')
plt.scatter(X_1[:, 0], X_1[:, 1], color='blue')

colors = np.array(['r', 'b'])
plt.scatter(test_points[:, 0], test_points[:, 1], color=colors[test_preds-1], alpha=0.25)
plt.scatter(X[:, 0], X[:, 1], color=colors[y-1])
plt.title("RBF-separated classes")

plt.show()
