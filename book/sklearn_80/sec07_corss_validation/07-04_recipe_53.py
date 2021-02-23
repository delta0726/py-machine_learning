# ******************************************************************************
# Chapter   : 7 交差検証とモデル構築後のフロー
# Title     : 7-4 均衡な交差検証（Recipe53)
# Created by: Owner
# Created on: 2020/12/28
# Page      : P206 - P208
# ******************************************************************************

# ＜概要＞
#


# ＜目次＞
# 0 準備
# 1 foldごとのループ処理


# 0 準備 ------------------------------------------------------------------------------------------

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


# データ生成
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 1, 1, 1, 2, 2, 2, 2])


# 1 foldごとのループ処理 -------------------------------------------------------------------

# ＜ポイント＞
# - 層化サンプリングされたFoldをさらにシャッフルすることが可能

# インスタンスの生成
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.25)
vars(sss)

# 反復処理をしてインデックスを作成
cc = 1
for train_index, test_index in sss.split(X, y):
    print('Round: ', cc, ": ",
          "Training indices: ", train_index,
          "Test indices :", test_index)
    cc += 1

# 分割数
sss.get_n_splits()
