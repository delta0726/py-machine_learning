# ******************************************************************************
# Chapter   : 6 距離尺度を使ったモデル構築
# Title     : 6-5 ミニバッチk-means法を使ってより多くのデータに対処する（Recipe45)
# Created by: Owner
# Created on: 2020/12/27
# Page      : P173 - P176
# ******************************************************************************

# ＜概要＞
# - kmeansは大規模データにおいては理想的な方法とは言えない
#   --- Scikit-Learnではkmeansは計算コストの高いアルゴリズム
#   --- ミニバッチk-means法を利用すれば、アルゴリズムの計算量を抑えたうえで近似解を得ることができる


# ＜目次＞
# 0 準備
# 1 ミニバッチk-meansとの比較


# 0 準備 ----------------------------------------------------------------------------------

import time
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise


# データ生成
# --- 100万レコードの大規模データ
blobs, labels = make_blobs(int(1e7), 3)


# 1 ミニバッチk-meansとの比較 ------------------------------------------------------------

# インスタンス生成
# --- k-means
kmeans = KMeans(n_clusters=3)
vars(kmeans)

# インスタンス生成
# --- ミニバッチk-means
minibatch = MiniBatchKMeans(n_clusters=3)
vars(minibatch)


# 学習
# --- k-means
# --- 約24.9sec
t1 = time.time()
kmeans.fit(blobs)
t2 = time.time()
print(t2 - t1)

# 学習
# --- ミニバッチk-means
# --- 約14.9sec
t1 = time.time()
minibatch.fit(blobs)
t2 = time.time()
print(t2 - t1)

# クラスタリングのセンター
# --- k-means
kmeans.cluster_centers_

# クラスタリングのセンター
# --- ミニバッチk-means
# --- 表示中所は異なっている可能性がある
minibatch.cluster_centers_

# 中心からの距離
# --- ペアワイズ距離
# --- 距離は非常に小さく近くにあることが分かる
pairwise.pairwise_distances(kmeans.cluster_centers_[0].reshape(1, -1),
                            minibatch.cluster_centers_[0].reshape(1, -1))

# ペアワイズ距離の対角成分
# --- クラスタの中心の差が設定される
np.diag(pairwise.pairwise_distances(kmeans.cluster_centers_,
                                    minibatch.cluster_centers_))
