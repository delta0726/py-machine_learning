# ******************************************************************************
# Title       : AI・データサイエンスのための数学プログラミング
# Chapter     : 2 機械学習を使った分析を行ってみよう
# Theme       : 2-3 大口顧客の類似性を主成分分析によって確認しよう
# Creat Date  : 2021/12/18
# Final Update:
# Page        : P66 - P68
# ******************************************************************************


# ＜概要＞
# - PCAで上位の成分の水準が似ている銘柄群は類似度が高いとみなすことができる
#   --- PCAを適用することで成分を効率的に比較することができる


# ＜目次＞
# 0 準備
# 1 特徴量ベクトルの抽出
# 2 主成分分析による可視化


# 0 準備 ---------------------------------------------------------------------------

# ライブラリ
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


# データ読み込み
df_info = pd.read_csv("csv/accomodation_info.csv", index_col=0, parse_dates=[0])

# データ確認
df_info


# 1 特徴量ベクトルの抽出 ------------------------------------------------------------

# ＜ポイント＞
# - 利用回数上位の顧客ごとの月間利用回数を特徴量ベクトルとする


# インデックスの取得
x_0 = df_info.resample('M')\
    .count()\
    .drop(df_info.columns.values, axis=1)

# 配列の準備
list_vector = []

# 人数の設定
num = 100

# 特徴量ベクトルの作成
# --- 顧客IDの抽出
# --- 月ごとの利用回数を特徴量として抽出
# --- 欠損値があった場合の穴埋め
# --- 特徴ベクトルとして追加
i_rank = 1
for i_rank in range(num):
    i_id = df_info['顧客ID'].value_counts().index[i_rank]
    x_i = df_info[df_info['顧客ID'] == i_id].resample('M').count()
    x_i = pd.concat([x_0, x_i], axis=1).fillna(0)
    list_vector.append(x_i.iloc[:, 0].values.tolist())

# 特徴量ベクトルの変換
features = np.array(list_vector)

# データ確認
print(features)


# 2 主成分分析による可視化 ------------------------------------------------------------

# ＜ポイント＞
# - PCAで主成分を抽出してPC1とPC2で2次元プロットを作成する
#   --- 上位成分の寄与度が大きいほどデータの類似性をうまく表現できる
#   --- データが集中している部分は類似度が高いサンプル


# 学習器の作成
pca = PCA()
pca.fit(features)

# 確認
vars(pca)

# 特徴ベクトルを主成分に変換
transformed = pca.fit_transform(features)

# 可視化
# --- PC1とPC2を抽出
for i in range(len(transformed)):
    plt.scatter(transformed[i, 0], transformed[i, 1], color="k")
    plt.text(transformed[i, 0], transformed[i, 1], str(i))

# プロット表示
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
