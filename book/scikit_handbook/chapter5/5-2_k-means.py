# ******************************************************************************
# Title     : Scikit-Learnデータ分析実装ハンドブック
# Chapter   : 5 クラスタリング
# Theme     : 5-2 k-means（非階層型クラスタリング）
# Created by: Owner
# Created on: 2021/5/18
# Page      : P232 - P239
# ******************************************************************************


# ＜概要＞
# - 非階層型クラスタリングの代表的手法であるk-meansでクラスタリングを実施する


# ＜目次＞
# 0 準備
# 1 データ基準化
# 2 モデル構築
# 3 クラスタ数ごとの可視化
# 4 正解ラベルとの比較
# 5 モデル出力の確認


# 0 準備 -------------------------------------------------------------------------------------

# ライブラリ
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine


# データロード
wine = load_wine()

# データ格納
# --- Xの特徴量は2つのみ（可視化ができるように）
X = wine.data[:, [9, 12]]
y = wine.target


# 1 データ基準化 ------------------------------------------------------------------------------

# インスタンス生成
sc = StandardScaler()

# データ変換
X_std = sc.fit_transform(X)

# データ確認
np.round(X_std.mean(axis=0))
np.round(X_std.std(axis=0))


# 2 モデル構築 --------------------------------------------------------------------------------

# インスタンス生成
model2 = KMeans(n_clusters=2, random_state=103)
model3 = KMeans(n_clusters=3, random_state=103)
model4 = KMeans(n_clusters=4, random_state=103)

# モデル訓練
model2.fit(X_std)
model3.fit(X_std)
model4.fit(X_std)


# 3 クラスタ数ごとの可視化 -------------------------------------------------------------------------

# ＜ポイント＞
# - 特徴量が2つのみ(2次元データ)なので散布図で可視化することが可能


#プロットのサイズ指定
plt.figure(figsize=(8, 12))

# K-Meansの散布図
# --- クラスタ数: 2
plt.subplot(3, 1, 1)
plt.scatter(X_std[:,0], X_std[:,1], c=model2.labels_)
plt.scatter(model2.cluster_centers_[:,0], model2.cluster_centers_[:,1],s=250, marker='*',c='red')
plt.title('K-means(n_clusters=2)')

# K-Meansの散布図
# --- クラスタ数: 3
plt.subplot(3, 1, 2)
plt.scatter(X_std[:,0], X_std[:,1], c=model3.labels_)
plt.scatter(model3.cluster_centers_[:,0], model3.cluster_centers_[:,1],s=250, marker='*',c='red')
plt.title('K-means(n_clusters=3)')

# K-Meansの散布図
# --- クラスタ数: 4
plt.subplot(3, 1, 3)
plt.scatter(X_std[:,0], X_std[:,1], c=model4.labels_)
plt.scatter(model4.cluster_centers_[:,0], model4.cluster_centers_[:,1],s=250, marker='*',c='red')
plt.title('K-means(n_clusters=4)')

plt.show()


# 4 正解ラベルとの比較 ----------------------------------------------------------------------------

# ＜ポイント＞
# - クラスタ数３が正解ラベルと似た結果になっているようだ
#   --- 目視で確認したもので、数値的な裏付けはない


#プロットのサイズ指定
plt.figure(figsize=(8, 8))

# 正解の散布図
# --- 色とプロリン
plt.subplot(2, 1, 1)
plt.scatter(X_std[:,0], X_std[:,1], c=y)
plt.title('training data y')

# K-Meansの散布図
plt.subplot(2, 1, 2)
plt.scatter(X_std[:,0], X_std[:,1], c=model3.labels_)
plt.scatter(model3.cluster_centers_[:,0], model3.cluster_centers_[:,1],s=250, marker='*',c='red')
plt.title('K-means(n_clusters=3)')

plt.show()


# 5 モデル出力の確認 -------------------------------------------------------------------------

# 分類結果
# --- 数値ラベルで表示される
model3.labels_

# k-meansの重心
model3.cluster_centers_
