# ******************************************************************************
# Chapter   : 8 サポートベクトルマシン
# Title     : 8-4 SVMによる多クラス分類（Recipe67)
# Created by: Owner
# Created on: 2020/12/30
# Page      : P252 - P255
# ******************************************************************************

# ＜概要＞
#


# ＜目次＞
# 0 準備
# 1 パイプラインの構築
# 2 クロスバリデーション


# 0 準備 ------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit
from itertools import product

# データロード
iris = datasets.load_iris()


# データ格納
X = iris.data[:, :2]
y = iris.target

X_0 = X[y == 0]
X_1 = X[y == 1]
X_2 = X[y == 2]

# データ分割
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)


# 1 パイプラインの構築 ----------------------------------------------------------------------

# インスタンス生成
svm_est = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', OneVsRestClassifier(SVC()))
])

# パラメータ設定
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1, 10]
param_grid = dict(svc__estimator__gamma=gammas, svc__estimator__C=Cs)


# 2 クロスバリデーション --------------------------------------------------------------------

# インスタンス生成
# --- クロスバリデーション(データ分割)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=7)

# インスタンス生成
# --- グリッドサーチ
rand_grid = RandomizedSearchCV(svm_est, param_distributions=param_grid,
                               cv=cv, n_iter=10)

# グリッドサーチの実行
rand_grid.fit(X_train, y_train)

# 最良パラメータ
rand_grid.best_params_


# 3 モデル可視化 --------------------------------------------------------------------------

# Minima and maxima of both features
xmin, xmax = np.percentile(X[:, 0], [0, 100])
ymin, ymax = np.percentile(X[:, 1], [0, 100])


# Grid/Cartesian product with itertools.product
test_points = np.array([[xx, yy] for xx, yy in product(np.linspace(xmin, xmax,100), np.linspace(ymin, ymax,100))])

# Predictions on the grid
test_preds = rand_grid.predict(test_points)

plt.figure(figsize=(15, 9))

plt.scatter(X_0[:, 0], X_0[:, 1], color='green')
plt.scatter(X_1[:, 0], X_1[:, 1], color='blue')
plt.scatter(X_2[:, 0], X_2[:, 1], color='red')

colors = np.array(['g', 'b', 'r'])
plt.tight_layout()
plt.scatter(test_points[:, 0], test_points[:, 1], color=colors[test_preds], alpha=0.25)
plt.scatter(X[:, 0], X[:, 1], color=colors[y])
plt.show()
