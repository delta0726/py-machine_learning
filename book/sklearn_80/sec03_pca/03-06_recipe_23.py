# ******************************************************************************
# Chapter   : 3 次元削減 - PCAから性能テストまで
# Title     : 3-6 分類のための分解にDictionaryLearningを使用する（Recipe23)
# Created by: Owner
# Created on: 2020/12/25
# Page      : P94 - P96
# ******************************************************************************

# ＜概要＞
#


# ＜目次＞
# 0 準備
# 1 辞書学習の実行
# 2 学習結果の可視化


# 0 準備 ------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.decomposition import DictionaryLearning

from mpl_toolkits.mplot3d import Axes3D


# データロード
iris = load_iris()

# データ格納
iris_X = iris.data
y = iris.target

# データ分割
X_train = iris_X[::2]
X_test = iris_X[1::2]
y_train = y[::2]
y_test = y[1::2]

# 確認
X_train.shape
X_test.shape


# 1 辞書学習の実行 --------------------------------------------------------------------------------

# インスタンス生成
# --- 3種類のアヤメの花を表すため3を指定
dl = DictionaryLearning(n_components=3)
vars(dl)

# 学習
# --- データ全体でなく訓練データで学習
transformed = dl.fit_transform(X_train)
transformed[:5]

# テストデータの変換
test_transform = dl.fit_transform(X_test)
test_transform[:5]


# 2 学習結果の可視化 --------------------------------------------------------------------------------

# プロット設定
fig = plt.figure(figsize=(14, 7))

# 訓練データ
ax = fig.add_subplot(121, projection='3d')
ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2],
           c=y_train, marker='^')
ax.set_title("Training Set")

# テストデータ
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(test_transform[:, 0], test_transform[:, 1], test_transform[:, 2],
           c=y_test, marker='^')
ax2.set_title("Testing Set")

# 表示
plt.show()
