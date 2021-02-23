# ******************************************************************************
# Chapter   : 6 距離尺度を使ったモデル構築
# Title     : 6-3 セントロイドの個数を最適化する（Recipe43)
# Created by: Owner
# Created on: 2020/12/27
# Page      : P166 - P169
# ******************************************************************************

# ＜概要＞
# - k-meansでは最適クラスタ数を事前に知ることはできない
#   --- ドメイン知識がない場合、基準を設定したうえで分析者が設定する必要がある
#   --- シルエット分析は中心からの観測値距離に着目した評価指標


# ＜目次＞
# 0 準備
# 1 ｋ-meansの実行
# 2 シルエット分析
# 3 プロット作成


# 0 準備 -----------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


# データ生成
blobs, classes = make_blobs(500, centers=3)


# 1 ｋ-meansの実行 ---------------------------------------------------------------------------

# インスタンス生成
kmean = KMeans(n_clusters=3)

# 学習
kmean.fit(blobs)
vars(kmean)


# 2 シルエット分析 ---------------------------------------------------------------------------

# シルエット距離の取得
# --- ｢(A)同じクラスタ内での非類似度｣と｢(B)最も近い別のクラスタの非類似度｣を元に計算
# --- A: 同じクラスタ内における、あるデータ点とその他すべてのデータ点との平均距離
# --- B: あるデータ点と最も近い別のクラスタの全データとの平均距離
silhuette_samples = metrics.silhouette_samples(blobs, kmean.labels_)
np.column_stack((classes[:5], silhuette_samples[:5]))

# データ確認
len(silhuette_samples)

# シルエットスコア
# --- シルエット距離の平均値
silhuette_samples.mean()

# シルエットスコア
# --- 関数で直接取得
metrics.silhouette_score(blobs, kmean.labels_)


# 3 プロット作成  ------------------------------------------------------------------------

# データ作成
blobs, classes = make_blobs(500, centers=10)

# シルエットスコア取得
silhuette_avgs = []

# ループ処理
# --- kmeansからシルエットスコアを計算
for k in range(2, 60):
    kmean = KMeans(n_clusters=k).fit(blobs)
    silhuette_avgs.append(metrics.silhouette_score(blobs, kmean.labels_))

# プロット作成
f, ax = plt.subplots(figsize=(7, 5))
ax.plot(silhuette_avgs)
plt.show()
