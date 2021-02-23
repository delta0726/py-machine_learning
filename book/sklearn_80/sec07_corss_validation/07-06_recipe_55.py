# ******************************************************************************
# Chapter   : 7 交差検証とモデル構築後のフロー
# Title     : 7-6 時系列交差検証（Recipe55)
# Created by: Owner
# Created on: 2020/12/28
# Page      : P211 - P212
# ******************************************************************************

# ＜概要＞
# - 時系列データでモデリングを行う際には時系列交差検証を行う


# ＜目次＞
# 0 準備
# 1 時系列交差検証


# 0 準備 ------------------------------------------------------------------------------------------

import numpy as np

from sklearn.model_selection import TimeSeriesSplit


# データ準備
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4],[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 1, 2, 3, 4])


# 1 時系列交差検証 --------------------------------------------------------------------------------

# インスタンス生成
tscv = TimeSeriesSplit(n_splits=7)
vars(tscv)

# データ分割
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("Train indices:", train_index, "Testing indices:", test_index)

# リストに出力
# --- イテレーションごとに配列がタプルに格納
# --- 全体をリストで格納
tscv_list = list(tscv.split(X))
tscv_list
