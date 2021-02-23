# ******************************************************************************
# Chapter   : 6 距離尺度を使ったモデル構築
# Title     : 6-4 クラスタの正確性を評価する（Recipe44)
# Created by: Owner
# Created on: 2020/12/27
# Page      : P169 - P173
# ******************************************************************************

# ＜概要＞
# -


# ＜目次＞
# 0 準備
# 1 ｋ-meansの実行
# 2 クラスタリングの性能評価


# 0 準備 ----------------------------------------------------------------------------------

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import cluster


# データ生成
blobs, ground_truth = datasets.make_blobs(1000, centers=3, cluster_std=1.75)

# プロット作成
f, ax = plt.subplots(figsize=(7, 5))
colors = ['r', 'g', 'b']
for i in range(3):
    p = blobs[ground_truth == i]
    ax.scatter(p[:, 0], p[:, 1], c=colors[i], label="Cluster {}".format(i))

ax.set_title("Cluster With Ground Truth")
ax.legend()
plt.show()


# 1 ｋ-meansの実行 -------------------------------------------------------------------------

# インスタンス生成
kmeans = cluster.KMeans(n_clusters=3)

# 学習
kmeans.fit(blobs)

# 結果確認
vars(kmeans).keys()

# クラスタの中央
kmeans.cluster_centers_

# プロット作成
# --- クラスタのセンターを追加
f, ax = plt.subplots(figsize=(7, 5))
colors = ['r', 'g', 'b']
for i in range(3):
    p = blobs[ground_truth == i]
    ax.scatter(p[:, 0], p[:, 1], c=colors[i], label="Cluster {}".format(i))
    ax.scatter(kmeans.cluster_centers_[:, 0],
               kmeans.cluster_centers_[:, 1],
               marker='*', s=250, color='black', label='Centers')

ax.set_title("Cluster With Ground Truth")
ax.legend()
plt.show()


# 2 クラスタリングの性能評価 -------------------------------------------------------------

#
for i in range(3):
    print((kmeans.labels_ == ground_truth)[ground_truth == i].astype(int).mean())

#
new_ground_truth = ground_truth.copy()
new_ground_truth[ground_truth == 1] = 2
new_ground_truth[ground_truth == 2] = 1


# 3 クラスタの慣性 -------------------------------------------------------------------

# ＜ポイント＞
# - 各データ点とそのデータが割り当てられたクラスタ中心との距離の二乗和
#   --- kmeansで最小化されている指標

# 慣性の取得
kmeans.inertia_
