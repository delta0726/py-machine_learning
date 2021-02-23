# ******************************************************************************
# Chapter   : 3 次元削減 - PCAから性能テストまで
# Title     : 3-7 次元削減に多様体学習(t-SNE)を使用する（Recipe24)
# Created by: Owner
# Created on: 2020/12/25
# Page      : P96 - P100
# ******************************************************************************

# ＜概要＞
# - irisデータセットを使った次元削減を複数の方法で行って比較する


# ＜目次＞
# 0 準備
# 1 4つの方法で変換
# 2 プロット


# 0 準備 ------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, Isomap


# データロード
iris = load_iris()

# データ格納
iris_X = iris.data
y = iris.target


# 1 4つの方法で変換 ---------------------------------------------------------------------------------

# ＜ポイント＞
# t-SNEは人気の高まっているアルゴリズムだが計算時間がかかる

# PCA
iris_pca = PCA(n_components=2).fit_transform(iris_X)

# tsne
iris_tsne = TSNE(learning_rate=200).fit_transform(iris_X)

# MDS
iris_MDS = MDS(n_components=2).fit_transform(iris_X)

# ISO
iris_ISO = Isomap(n_components=2).fit_transform(iris_X)


# 2 プロット --------------------------------------------------------------------------------------

# プロット設定
plt.figure(figsize=(20, 10))

# PCA
plt.subplot(221)
plt.scatter(iris_pca[:, 0], iris_pca[:, 1], c=y)
plt.title('PCA')

# tsne
plt.subplot(222)
plt.scatter(iris_tsne[:, 0], iris_tsne[:, 1], c=y)
plt.title('TSNE')

# MDS
plt.subplot(223)
plt.scatter(iris_MDS[:, 0], iris_MDS[:, 1], c=y)
plt.title('MDS')

# ISO
plt.subplot(224)
plt.scatter(iris_ISO[:, 0], iris_ISO[:, 1], c=y)
plt.title('ISO')

# 表示
plt.show()
