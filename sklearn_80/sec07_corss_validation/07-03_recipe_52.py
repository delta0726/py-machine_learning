# ******************************************************************************
# Chapter   : 7 交差検証とモデル構築後のフロー
# Title     : 7-3 k分割交差検証（Recipe52)
# Created by: Owner
# Created on: 2020/12/28
# Page      : P204 - P206
# ******************************************************************************

# ＜概要＞
# - 交差検証の各Foldのインデックスを可視化することで、Foldに含まれるデータを可視化する


# ＜目次＞
# 0 準備
# 1 foldごとのループ処理
# 2 インデックスから要素を抽出


# 0 準備 ------------------------------------------------------------------------------------------

import numpy as np

from sklearn.model_selection import KFold


# データ生成
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 1, 2, 1, 2, 1, 2])


# 1 foldごとのループ処理 ------------------------------------------------------------------

# ＜ポイント＞
# - インデックスで訓練データとテストデータを区別している
#   --- 元データの要素自体を表示していない点に注意


# インスタンス生成
kf = KFold(n_splits=4)
vars(kf)

# 反復処理をしてインデックスを作成
cc = 1
for train_index, test_index in kf.split(X):
    print('Round: ', cc, ": ",
          "Training indices: ", train_index,
          "Test indices :", test_index)
    cc += 1

# 分割数
kf.get_n_splits()


# 2 インデックスから要素を抽出 ------------------------------------------------------------

# Fold自体のデータを可視化
# --- リストに出力
indices_list = list(kf.split(X))
indices_list

# タプルを分割
# --- タプルにnumpy配列が格納されている（分割用のインデックス）
train_indices, test_indices = indices_list[3]

# 学習データ(X)の確認
# --- 訓練データ / テストデータ
X[train_indices]
X[test_indices]

# ラベルデータ(y)の確認
# --- 訓練データ / テストデータ
y[train_indices]
y[test_indices]
