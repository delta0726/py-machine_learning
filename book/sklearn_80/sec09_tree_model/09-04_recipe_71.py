# ******************************************************************************
# Chapter   : 9 決定木アルゴリズムとアンサンブル学習
# Title     : 9-4 決定木のチューニング（Recipe71)
# Created by: Owner
# Created on: 2020/12/28
# Page      : P262 - P272
# ******************************************************************************

# ＜概要＞
#


# ＜目次＞
# 0 準備
# 1 モデリング
# 2 グリッドサーチ
# 3 グリッドサーチの結果検証


# 0 準備 --------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# データロード
iris = load_iris()

# データ格納
X = iris.data[:, :2]
y = iris.target

# データ分割
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, stratify=y)

# データフレームで表示
pd.DataFrame(X, columns=iris.feature_names[:2])

# 訓練データの可視化
plt.figure(figsize=(12, 6))
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.show()


# 1 モデリング ---------------------------------------------------------------------

# インスタンス生成
# --- デフォルトのパラメータを使用
dtc = DecisionTreeClassifier()
vars(dtc)

# 学習
dtc.fit(X_train, y_train)
vars(dtc)

# 予測
y_pred = dtc.predict(X_test)

# モデル精度
accuracy_score(y_true=y_test, y_pred=y_pred)


# 2 グリッドサーチ ---------------------------------------------------------------------

# インスタンス生成
# --- 決定木の分類推定器
dtc = DecisionTreeClassifier()

# パラメータ設定
# --- グリッドサーチ
param_grid = {'criterion': ['gini', 'entropy'],
              'max_depth': [3, 5, 7, 20]}

# インスタンス生成
# --- グリッドサーチ・クロスバリデーション
gs_inst = GridSearchCV(dtc, param_grid=param_grid)
vars(gs_inst)

# 実行
# --- グリッドサーチ・クロスバリデーション
gs_inst.fit(X_train, y_train)
vars(gs_inst)

# 予測値の取得
y_pred_gs = gs_inst.predict(X_test)

# モデル精度
accuracy_score(y_true=y_test, y_pred=y_pred_gs)


# 3 グリッドサーチの結果検証 ----------------------------------------------------------

# 予測精度の平均値
means = gs_inst.cv_results_['mean_test_score']
means

# 予測精度の標準偏差
stds = gs_inst.cv_results_['std_test_score']
stds

# パラメータ
params = gs_inst.cv_results_['params']
params

# ループ取得
for mean, std, param in zip(means, stds, params):
    print("mean: %0.5f, std: %0.5f, params: %r" % (mean, std, param))


# ベストモデル
gs_inst.best_estimator_

# 可視化
# 省略


# 4 散布図における決定木の可視化 ----------------------------------------------------------

# 再現できず

# # パラメータ設定
# grid_interval = 0.02
#
#
# x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
# y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
#
# xmin, xmax = np.percentile(X[:, 0], [0, 100])
# ymin, ymax = np.percentile(X[:, 1], [0, 100])
#
# xmin_plot, xmax_plot = xmin - .5, xmax + .5
# ymin_plot, ymax_plot = ymin - .5, ymax + .5
#
# xx, yy = np.meshgrid(np.arange(xmin_plot, xmax_plot, grid_interval),
#                      np.arange(ymin_plot, ymax_plot, grid_interval))
#
#
# X_0 = X[y == 0]
# X_1 = X[y == 1]
# X_2 = X[y == 2]
#
# plt.figure(figsize=(15,8))
# plt.scatter(X_0[:, 0], X_0[:, 1], color='red')
# plt.scatter(X_1[:, 0], X_1[:, 1], color='blue')
# plt.scatter(X_2[:, 0], X_2[:, 1], color='green')
#
# test_preds = gs_inst.best_estimator_.predict(np.array(zip(xx.ravel(), yy.ravel())))
#
# colors = np.array(['r', 'b', 'g'])
# plt.scatter(xx.ravel(), yy.ravel(), color=colors[test_preds], alpha=0.15)
# plt.scatter(X[:, 0], X[:, 1], color=colors[y])
# plt.title("Decision Tree Visualization")
# plt.xlabel(iris.feature_names[0])
# plt.ylabel(iris.feature_names[1])
# plt.show()


