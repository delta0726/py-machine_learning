# ******************************************************************************
# Title       : AI・データサイエンスのための数学プログラミング
# Chapter     : 2 機械学習を使った分析を行ってみよう
# Theme       : 2-1 顧客の行動パターンの類似度を計算しよう
# Creat Date  : 2021/12/18
# Final Update: 2022/9/2
# Page        : P60 - P63
# ******************************************************************************


# ＜概要＞
# - 特徴量を数値で表現することで類似度を計算する
#   --- 類似度分析のイントロダクションなのでインプリケーションは少ない


# ＜目次＞
# 0 準備
# 1 特徴ベクトルの可視化
# 2 類似度の計算


# 0 準備 ---------------------------------------------------------------------------

# ライブラリ
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# データ読み込み
# --- 日付列をインデックスにする
df_info = pd.read_csv("csv/accomodation_info.csv", index_col=0, parse_dates=[0])

# データ確認
df_info


# 1 特徴ベクトルの可視化 ------------------------------------------------------------

# ＜ポイント＞
# - 月ごとの顧客の利用回数を特徴量として顧客ごとの類似度を確認する
#   --- 時系列の折れ線グラフを重ね合わせて類似度を確認する


# indexの抽出
# --- 月末日付のインデックス
# --- インデックスのみのDataFrameを作成
x_0 = df_info.resample('M').count()
x_0 = x_0.drop(x_0.columns.values, axis=1)
type(x_0)

# 順位の設定
i_rank = 1
j_rank = 2

# 顧客IDの抽出
i_id = df_info['顧客ID'].value_counts().index[i_rank]
j_id = df_info['顧客ID'].value_counts().index[j_rank]

# 月次カウントの取得
# --- 月ごとの利用回数を特徴量として抽出
x_i = df_info.loc[lambda x: x['顧客ID'] == i_id].resample('M').count()
x_j = df_info.loc[lambda x: x['顧客ID'] == j_id].resample('M').count()

# データ結合
# --- 欠損値があった場合の穴埋め
x_i = pd.concat([x_0, x_i], axis=1).fillna(0)
x_j = pd.concat([x_0, x_j], axis=1).fillna(0)

# プロット作成
# --- 系列の連動性から類似度を確認する
plt.plot(x_i)
plt.plot(x_j)
plt.xticks(rotation=60)
plt.show()


# 2 類似度の計算 -------------------------------------------------------------------

# ＜ポイント＞
# - 距離を次元数で割ったものを非類似度として定義する
#   --- 数値が小さいほど類似度が高いため｢非類似度｣という


# 特徴ベクトルの差を計算
# --- ランク1とランク2の距離
dx = x_i.iloc[:, 0].values - x_j.iloc[:, 0].values

# 距離の計算
# --- ユークリッド距離の算出
n = np.linalg.norm(dx)

# 次元による正規化
d = n / len(x_i)
print("類似度:", d)
