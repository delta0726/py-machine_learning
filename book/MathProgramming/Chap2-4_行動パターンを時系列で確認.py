# ******************************************************************************
# Title       : AI・データサイエンスのための数学プログラミング
# Chapter     : 2 機械学習を使った分析を行ってみよう
# Theme       : 2-4 大口顧客の行動パターンを時系列によって確認
# Creat Date  : 2021/12/20
# Final Update:
# Page        : P69 - P71
# ******************************************************************************


# ＜概要＞
# - PCAで近くにいるサンプルは類似した動きをすることが期待される
#   --- 顧客ごとの時系列グラフで確認


# ＜目次＞
# 0 準備
# 1 特徴量ベクトルのプロット
# 2 PCが近いIDのプロット


# 0 準備 -------------------------------------------------------------------

# ライブラリ
import pandas as pd
import matplotlib.pyplot as plt


# データ読み込み
df_info = pd.read_csv("csv/accomodation_info.csv", index_col=0, parse_dates=[0])

# データ確認
df_info


# 1 特徴量ベクトルのプロット --------------------------------------------------

# ＜ポイント＞
# - 月ごとの顧客の利用回数を特徴量とする
# - 特徴量を顧客ごとに定義する（行：日時 列：顧客ID）


# indexの抽出
x_0 = df_info.resample('M').count()
x_0 = x_0.drop(x_0.columns.values, axis=1)

# パラメータ設定
list_rank = [0, 1, 2]

# プロット作成
# --- 指定した顧客IDの抽出
# --- 特徴量ベクトルの作成（月次の利用回数）
# --- 系列をプロット
for i_rank in list_rank:
    i_id = df_info['顧客ID'].value_counts().index[i_rank]
    x_i = df_info.loc[lambda x: x['顧客ID'] == i_id]\
        .resample('M').count()\
        .filter(['顧客ID'])\
        .fillna(0)
    plt.plot(x_i)
    plt.xticks(rotation=20)

# プロット表示
plt.show()


# 2 PCが近いIDのプロット -----------------------------------------------------

# ＜ポイント＞
# - Chap2-2でのPCAの2次元プロットで近くにいたサンプルでプロットを作成
#   --- 時系列推移が類似していることが期待される


# 関数定義
# --- 1で定義した処理を関数化
def plot_ts(list_rank):
    for i_rank in list_rank:
        i_id = df_info['顧客ID'].value_counts().index[i_rank]
        x_i = df_info.loc[lambda x: x['顧客ID'] == i_id] \
            .resample('M').count() \
            .filter(['顧客ID']) \
            .fillna(0)
        plt.plot(x_i)
        plt.xticks(rotation=20)

    # プロット表示
    plt.show()


# プロット作成
list_rank = [22, 25, 42]
plot_ts(list_rank)

# プロット作成
list_rank = [49, 64, 70]
plot_ts(list_rank)
