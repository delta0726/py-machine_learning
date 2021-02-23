# ******************************************************************************
# Chapter   : 6 距離尺度を使ったモデル構築
# Title     : 6-2 k-means法を使ったデータのクラスタリング（Recipe42)
# Created by: Owner
# Created on: 2020/12/27
# Page      : P162 - P166
# ******************************************************************************

# ＜概要＞
# - k-meansによるクラスタリングを実行する
# - kmeansはクラスタ内の各観測値の平均と観測点の二乗距離の合計を最小化してくアルゴリズム


# ＜目次＞
# 0 準備
# 1 k-meansの実行


# 0 準備 ------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


# データ生成
blobs, classes = make_blobs(500, centers=3)

# プロット作成
# --- 生成データ
f, ax = plt.subplots(figsize=(7.5, 7.5))
rgb = np.array(['r', 'g', 'b'])
ax.scatter(blobs[:, 0], blobs[:, 1], color=rgb[classes])
ax.set_title("Blobs")
plt.show()


# 1 k-meansの実行 -----------------------------------------------------------------------------

# インスタンス生成
kmeans = KMeans(n_clusters=3)
vars(kmeans)

# 学習
kmeans.fit(X=blobs)
vars(kmeans)

# 各クラスタの中心
kmeans.cluster_centers_

# プロット作成
f, ax = plt.subplots(figsize=(7.5, 7.5))
rgb = np.array(['r', 'g', 'b'])
ax.scatter(blobs[:, 0], blobs[:, 1], color=rgb[classes])
ax.scatter(kmeans.cluster_centers_[:, 0],
           kmeans.cluster_centers_[:, 1],
           marker='*', s=250, color='black', label='Centers')
ax.set_title("Blobs")
plt.show()

# 期待ラベル
# --- 観測値ごと
kmeans.labels_[:10]

# 注意事項
# --- 元データにはクラスがあるが、k-meansはこれを学習したわけではない
# --- k-meansから出力されたラベルの出力値と元クラスの出力値は一致していない（関係性は近くなっている）
classes[:10]

# 中心からの距離を取得
kmeans.transform(blobs)[:5]
dir(kmeans)
