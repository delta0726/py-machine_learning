# ******************************************************************************
# Chapter   : 2 モデル構築前のワークフローと前処理
# Title     : 2-2 簡単な分析を行うためのサンプルデータを作成する（Recipe10)）
# Created by: Owner
# Created on: 2020/12/23
# Page      : P43 - P45
# ******************************************************************************

# ＜概要＞
# - sklearnの中にはirisのような既存データセット以外に、目的に応じてデータセットを作る関数がある
#   --- dataset.make_*


# ＜目次＞
# 0 準備
# 1 回帰データセットの作成
# 2 不均衡な分類データセットの作成
# 3 クラスタリングのためのデータセット


# 0 準備 --------------------------------------------------------------------------------------

import sklearn.datasets as d
import matplotlib.pyplot as plt


# 関数一覧
# d.make_*?


# 1 回帰データセットの作成 -----------------------------------------------------------------------

# 回帰用のデータ作成
# --- 100行×100列
reg_data = d.make_regression()

# 確認
len(reg_data)
reg_data[0]
reg_data[1]

# 回帰用の複雑なデータ作成
# --- 情報利得を持つ特徴量：5
# --- 目的変数の値：2
# --- バイアス項：1
complex_reg_data = d.make_regression(n_samples=1000, n_features=10, n_informative=5,
                                     n_targets=2, bias=1.0)

# 確認
len(complex_reg_data)
complex_reg_data[0].shape
complex_reg_data[1].shape
complex_reg_data[0][0:10, :]
complex_reg_data[1][0:10, :]


# 2 不均衡な分類データセットの作成 -----------------------------------------------------------------------

# 分類データの作成
classification_set = d.make_classification(n_samples=100, n_features=20, weights=[0.1])

# 確認
len(classification_set)
classification_set[0]
classification_set[1]

# ラベルの割合
classification_set[1].sum() / len(classification_set[1])


# 3 クラスタリングのためのデータセット --------------------------------------------------------------------

# 等方向のガウス分布データ
blobs_data, blobs_target = d.make_blobs()

# プロット作成
plt.scatter(x=blobs_data[:, 0], y=blobs_data[:, 1], c=blobs_target)
plt.show()
