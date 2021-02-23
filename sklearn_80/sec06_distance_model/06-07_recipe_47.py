# ******************************************************************************
# Chapter   : 6 距離尺度を使ったモデル構築
# Title     : 6-7 特徴空間において最近傍を特定する（Recipe47)
# Created by: Owner
# Created on: 2020/12/27
# Page      : P179 - P184
# ******************************************************************************

# ＜概要＞
# -


# ＜目次＞
# 0 準備
# 1 オブジェクトの近接性


# 0 準備 ------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise
from sklearn.datasets import make_blobs


# データ生成
points, labels = make_blobs()


# 1 距離行列 --------------------------------------------------------------------------

# 距離行列の作成
# --- ペアワイズ距離
distances = pairwise.pairwise_distances(points)
distances

# 対角成分
# --- 距離行列の対角成分は全てゼロ
np.diag(distances)[:5]

# 要素[0]から見た距離
distances[0][:5]

# 近接性をランク化
ranks = np.argsort(distances[0])
ranks

# ランクで並べた際の元データ
sp_points = points[ranks][:5]
sp_points


# 2 プロット作成 --------------------------------------------------------------------

# プロット作成
plt.figure(figsize=(10, 7))
plt.scatter(points[:, 0], points[:, 1], label='All Points')
plt.scatter(sp_points[:, 0], sp_points[:, 1], color='red',
            label='Closest Points')
plt.scatter(points[0, 0], points[0, 1], color='green', label='Chosen Point')
plt.legend()
plt.show()

