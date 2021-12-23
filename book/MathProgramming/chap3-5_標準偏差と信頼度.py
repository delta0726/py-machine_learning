# ******************************************************************************
# Title       : AI・データサイエンスのための数学プログラミング
# Chapter     : 3 必要なデータ数を検討しよう
# Theme       : 3-5 標準偏差と信頼度の関係
# Creat Date  : 2021/12/24
# Final Update:
# Page        : P109 - P110
# ******************************************************************************


# ＜概要＞
# - 信頼度は標準正規分布における信頼区間の面積の割合のことをいう
# - 信頼度は標準偏差の設定によって水準が変わってくる


# ＜目次＞
# 0 準備
# 1 母集団の生成
# 2 サンプリング平均
# 3 信頼度の計算


# 0 準備 ---------------------------------------------------------------------

# ライブラリ
import numpy as np
import matplotlib.pyplot as plt


# 1 母集団の生成 -------------------------------------------------------------

# ＜ポイント＞
# - 乱数により母集団を生成する


# パラメータ設定
# --- 母集合の大きさ
# --- 乱数の平均値
# --- 乱数の標準偏差
num = 365 * 2
ave = 0.0
std = 1.0

# 乱数を発生（シードを固定）
np.random.seed(seed=0)
x = np.random.normal(ave, std, num)
# x = np.random.exponential(0.5, num)

# 統計量の計算
# --- 母平均
# --- 母標準偏差
x_ave = np.average(x)
x_std = np.std(x)

# 確認
print("母平均:", x_ave)
print("母標準偏差:", x_std)

# プロット作成
num_bin = 21
plt.hist(x, num_bin, color="k")
plt.xlim([-5, 5])
plt.show()


# 2 サンプリング平均 --------------------------------------------------

# ＜ポイント＞
# - サンプリング結果の平均値は正規分布を描く


# パラメータ設定
# --- サンプリング数
# --- シミュレーション回数を設定
num_sample = 30
num_trial = 10000

# 配列生成
x_trial = np.zeros(num_trial)

# サンプル平均の算出を試行
for i in range(num_trial):
    x_sample = np.random.choice(x, num_sample)
    x_ave = np.average(x_sample)
    x_trial[i] = x_ave

# 統計量の計算
# --- サンプル平均
# --- サンプル標準偏差
x_trial_ave = np.average(x_trial)
x_trial_std = np.std(x_trial)

# 確認
print("サンプル平均:", x_trial_ave)
print("サンプル標準偏差:", x_trial_std)

# プロット
num_bin = 21
plt.hist(x_trial, num_bin, color="k")
plt.xlim([-5, 5])
plt.show()


# 3 信頼度の計算 --------------------------------------------------------

# ＜ポイント＞
# - 正規分布の信頼区間の面積を信頼度という


# パラメータ設定
# --- 標準偏差の倍率(σ)
ratio = 1.0

# 信頼区間の計算
# --- 左側の領域外の割合の計算
# --- 右側の領域外の割合の計算
x_trial_out1 = x_trial[x_trial > x_trial_ave + ratio * x_trial_std]
x_trial_out2 = x_trial[x_trial < x_trial_ave - ratio * x_trial_std]

# 信頼度の計算
# --- 信頼区間の面積の割合
reliability = 1 - (len(x_trial_out1) / len(x_trial) + len(x_trial_out2) / len(x_trial))
print("信頼度:", reliability)
