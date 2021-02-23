# ******************************************************************************
# Chapter   : 6 距離尺度を使ったモデル構築
# Title     : 6-6 k-means法による画像の量子化（Recipe46)
# Created by: Owner
# Created on: 2020/12/27
# Page      : P176 - P179
# ******************************************************************************

# ＜概要＞
#


# ＜目次＞
# 0 準備


# 0 準備 ------------------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster


# パスの取得
current_path = os.getcwd()
file = os.path.sep.join(['', 'picture', 'headshot.jpg'])

# 画像の読み込み
img = plt.imread(current_path + file)
plt.figure(figsize=(10, 7))
plt.imshow(img)
plt.show()


# 1 画像の量子化 -----------------------------------------------------------------------

# 次元数の確認
# --- 3次元のデータであることが分かる
img.shape

# 配列を2次元に変換
# --- 画像を量子化するには2次元配列に変換する必要がある
x, y, z = img.shape
long_img = img.reshape(x * y, z)
long_img.shape


# 2 クラスタリングの適用 ----------------------------------------------------------------

# インスタンスの生成
k_means = cluster.KMeans(n_clusters=5)

# 学習
k_means.fit(long_img)

# センターの確認
centers = k_means.cluster_centers_
centers

# ラベルの取得
labels = k_means.labels_
labels

# 量子化された画像の表示
plt.figure(figsize=(10, 7))
plt.imshow(centers[labels].reshape(x, y, z).astype(np.uint8))
plt.show()
