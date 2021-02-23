# ******************************************************************************
# Chapter   : 8 サポートベクトルマシン
# Title     : 8-2 線形SVMを使ってデータを分類する（Recipe65)
# Created by: Owner
# Created on: 2020/12/30
# Page      : P240 - P246
# ******************************************************************************

# ＜概要＞
#


# ＜目次＞
# 0 準備
# 1 モデリング
# 2 モデル評価
# 3 モデル可視化
# 4 モデル確認

# 0 準備 ------------------------------------------------------------------------------------------

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# データロード
iris = datasets.load_iris()

# データ格納
X_w = iris.data[:, :2]
y_w = iris.target

X = X_w[y_w < 2]
y = y_w[y_w < 2]

# プロット作成
X_0 = X[y == 0]
X_1 = X[y == 1]
plt.figure(figsize=(10, 7))
plt.scatter(X_0[:, 0], X_0[:, 1], color='red')
plt.scatter(X_1[:, 0], X_1[:, 1], color='blue')
plt.show()

# データ分割
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)


# 1 モデリング -------------------------------------------------------------------------------

# インスタンス生成
svm_inst = SVC(kernel='linear')

# 学習
svm_inst.fit(X_train, y_train)


# 2 モデル評価 -------------------------------------------------------------------------------

# 予測
y_pred = svm_inst.predict(X_test)

# Accuracy
accuracy_score(y_test, y_pred)


# 3 モデル可視化 ---------------------------------------------------------------------------

# Minima and maxima of both features
xmin, xmax = np.percentile(X[:, 0], [0, 100])
ymin, ymax = np.percentile(X[:, 1], [0, 100])

# Grid/Cartesian product with itertools.product
test_points = np.array([[xx, yy] for xx, yy in product(np.linspace(xmin, xmax), np.linspace(ymin, ymax))])

# Predictions on the grid
test_preds = svm_inst.predict(test_points)


X_0 = X[y == 0]
X_1 = X[y == 1]

plt.figure(figsize=(10, 7))
plt.scatter(X_0[:, 0], X_0[:, 1], color='red')
plt.scatter(X_1[:, 0], X_1[:, 1], color='blue')

colors = np.array(['r', 'b'])
plt.scatter(test_points[:, 0], test_points[:, 1], color=colors[test_preds], alpha=0.25)
plt.scatter(X[:, 0], X[:, 1], color=colors[y])
plt.title("Linearly-separated classes")
plt.show()


# 4 モデル確認 ----------------------------------------------------------------------------

# 法線ベクトル
svm_inst.coef_

# 切片
svm_inst.intercept_

# パラメータ
svm_inst
