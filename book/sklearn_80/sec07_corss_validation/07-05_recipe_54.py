# ******************************************************************************
# Chapter   : 7 交差検証とモデル構築後のフロー
# Title     : 7-5 Shuffle Splitによる交差検証（Recipe54)
# Created by: Owner
# Created on: 2020/12/28
# Page      : P208 - P210
# ******************************************************************************

# ＜概要＞
# - ｢ShuffleSplit｣は単純な交差検証のためのクラス


# ＜目次＞
# 0 準備
# 1 サンプリングと平均値
# 2 ShuffleSplitクラス


# 0 準備 ------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import ShuffleSplit


# データ作成
true_mean = 1000
true_std = 10
N = 1000
dataset = np.random.normal(loc=true_mean, scale=true_std, size=N)

# データの可視化
f, ax = plt.subplots(figsize=(10, 7))
ax.hist(dataset, color='k', alpha=0.65, histtype='stepfilled', bins=50)
ax.set_title('Histogram of dataset')
plt.show()


# 1 サンプリングと平均値 ---------------------------------------------------------------------------

# データ分割
# - データセットのうち半分で分割
holdout_set = dataset[:500]
fitting_set = dataset[500:]

# 平均値の算出
# --- 分割したデータセット
estimate = fitting_set.mean()
estimate

# 平均値の算出
# --- 分割したデータセット
data_mean = dataset.mean()
data_mean


# 2 ShuffleSplitクラス ------------------------------------------------------------------------

# インスタンス生成
shuffle_split = ShuffleSplit(n_splits=100, test_size=0.5, random_state=0)
vars(shuffle_split)

# オブジェクト定義
mean_p = []
estimate_closeness = []

# シミュレーション
for train_index, not_used_index in shuffle_split.split(fitting_set):
    mean_p.append(fitting_set[train_index].mean())
    shuf_estimate = np.mean(mean_p)
    estimate_closeness.append(np.abs(shuf_estimate - dataset.mean()))


# プロット作成
plt.figure(figsize=(10, 5))
plt.plot(estimate_closeness)
plt.show()
