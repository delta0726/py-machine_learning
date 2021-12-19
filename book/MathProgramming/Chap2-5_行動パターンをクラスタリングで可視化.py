# ******************************************************************************
# Title       : AI・データサイエンスのための数学プログラミング
# Chapter     : 2 機械学習を使った分析を行ってみよう
# Theme       : 2-5 大口顧客同士の行動パターンの違いをクラスタリングによって可視化
# Creat Date  : 2021/12/18
# Final Update:
# Page        : P72 - P75
# ******************************************************************************


# ＜概要＞
# - PCAとクラスタリングを組み合わせることで、データ類似性を効率的に可視化する


# ＜目次＞
# 0 準備
# 1 特徴量ベクトルの作成
# 2 k-meansによるクラスタリング
# 3 PCAによる可視化


# 0 準備 ---------------------------------------------------------------------------

# ライブラリ
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# データ読み込み
df_info = pd.read_csv("csv/accomodation_info.csv", index_col=0, parse_dates=[0])

# データ確認
df_info


# 1 特徴量ベクトルの作成 ------------------------------------------------------------

# ＜ポイント＞
# - 利用回数上位の顧客ごとの月間利用回数を特徴量ベクトルとする


# インデックスの取得
x_0 = df_info.resample('M')\
    .count()\
    .drop(df_info.columns.values, axis=1)

# パラメータ設定
# --- 対象人数の設定
num = 100

# 配列の準備
list_vector = []

# 特徴量ベクトルの作成
# --- 顧客IDの抽出
# --- 月ごとの利用回数を特徴量として抽出
# --- 欠損値があった場合の穴埋め
# --- 特徴ベクトルとして追加
for i_rank in range(num):
    i_id = df_info['顧客ID'].value_counts().index[i_rank]
    x_i = df_info.loc[lambda x: x['顧客ID'] == i_id] \
        .resample('M') \
        .count() \
        .filter(['顧客ID'])
    x_i = pd.concat([x_0, x_i], axis=1).fillna(0)
    list_vector.append(x_i.iloc[:, 0].values.tolist())

# 特徴量ベクトルの変換
features = np.array(list_vector)

# データ確認
print(features)


# 2 k-meansによるクラスタリング -----------------------------------------------------

# ＜ポイント＞
# - 分割型クラスタリング(k-means)でクラス分類を行う


# パラメータ設定
# --- クラスタ数
num_of_cluster = 4

# モデル構築
model = KMeans(n_clusters=num_of_cluster, random_state=0)
model.fit(features)

# モデル確認
vars(model)

# クラスタの予測
pred_class = model.predict(features)

# クラス確認
pred_class


# 3 PCAによる可視化 ----------------------------------------------------------------

# ＜ポイント＞
# - PCAにより2次元プロットで可視化してクラスタリングで色分けする
#   --- PCAとクラスタリングの組み合わせでプロットするのは鉄板


# モデル構築
pca = PCA()
pca.fit(features)

# モデル確認
vars(pca)

# データ変換
transformed = pca.fit_transform(features)

# プロット作成
# --- PC1とPC2で散布図を作成（クラスタで色分け）
# --- 系列表示
plt.scatter(transformed[:, 0], transformed[:, 1], c=pred_class)

for i in range(len(transformed)):
    text = str(i) + "(" + str(pred_class[i]) + ")"
    plt.text(transformed[i, 0], transformed[i, 1], text)

plt.show()
