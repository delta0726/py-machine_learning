# ******************************************************************************
# Title       : AI・データサイエンスのための数学プログラミング
# Chapter     : 3 必要なデータ数を検討しよう
# Theme       : 3-4 1ヵ月分のデータから二年分のデータの平均・標準偏差を推定
# Creat Date  : 2021/12/24
# Final Update:
# Page        : P105 - P108
# ******************************************************************************


# ＜概要＞
# - 実際のデータにおけるサンプリング平均の標準偏差と中心極限定理を見比べる
#   --- 実際のデータにバラツキが大きくサンプル数も少ないので、標準偏差の推定値は大きく乖離する


# ＜目次＞
# 0 準備
# 1 データ整理
# 2 ランダムサンプリング
# 3 中心極限定理から母集団の標準偏差を逆推定


# 0 準備 ---------------------------------------------------------------------

# ライブラリ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# データ取得
df_theft_201811 = pd.read_csv("csv/theft_list_201811.csv", index_col=0, parse_dates=[0])
df_amenity_price = pd.read_csv("csv/amenity_price.csv", index_col=0, parse_dates=[0])


# 1 データ整理 ----------------------------------------------------------------

# 配列作成（日数分）
list_amount = np.zeros(len(df_theft_201811.index))

# 被害額をリストに格納
for i_index in range(len(df_theft_201811.index)):
    for i_column in range(len(df_theft_201811.columns)):
        list_amount[i_index] += df_theft_201811.iloc[i_index, i_column] * \
                                df_amenity_price["金額"].iloc[i_column]

# プロット作成
plt.plot(list_amount, color="k")
plt.show()


# 2 ランダムサンプリング ------------------------------------------------------

# パラメータ設定
# --- サンプル数（サンプル集団の大きさ）
# --- 試行回数を設定
num_sample = 10
num_trial = 10000

# 配列生成
x_trial = np.zeros(num_trial)

# サンプル平均の算出を試行
for i in range(num_trial):
    x_sample = np.random.choice(list_amount, num_sample)
    x_ave = np.average(x_sample)
    x_trial[i] = x_ave

# 統計量の確認
# --- サンプル平均の平均
# --- サンプル平均の標準偏差
x_trial_ave = np.average(x_trial)
x_trial_std = np.std(x_trial)

# 確認
print("平均:", x_trial_ave)
print("標準偏差:", x_trial_std)

# 描画
num_bin = 21
plt.hist(x_trial, num_bin, color="k")
plt.xlim([-50000, 50000])
plt.show()


# 3 中心極限定理から母集団の標準偏差を逆推定 ---------------------------------------

# パラメータ設定
# --- 標準偏差
# --- サンプル数
sample_std = 5649
num_sample = 10

# 母集団の分散を計算
org_std = np.sqrt(num_sample)*sample_std
print("母集団の標準偏差:", org_std)
