# ******************************************************************************
# Chapter   : 2 モデル構築前のワークフローと前処理
# Title     : 2-3 標準分布の尺度でデータをスケーリング（Recipe11)
# Created by: Owner
# Created on: 2020/12/23
# Page      : P46 - P51
# ******************************************************************************

# ＜概要＞
# - 前処理の基本としてデータ基準化がある
#   --- sklearnでは関数やクラスで操作が可能


# ＜目次＞
# 0 準備
# 1 データ操作
# 2 可視化
# 3 クラスを用いた基準化
# 4 Max-Min Scaling


# 0 準備 ---------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.datasets import load_boston


# データ準備
boston = load_boston()

# データ概要
boston.keys()
boston.items()

# 変数定義
X, y = boston.data, boston.target


# 1 データ操作 -----------------------------------------------------------------------------

# データ確認
X[:, :3]

# 平均値
X[:, :3].mean(axis=0)

# 標準偏差
X[:, :3].std(axis=0)

# 基準化
# --- 関数による変換
X_2 = preprocessing.scale(X[:, :3])
X_2.mean(axis=0)
X_2.std(axis=0)


# 2 可視化 -----------------------------------------------------------------------------

# ヒストグラムの作成
# --- 元データ
pd.Series(X[:, 2]).hist(bins=50)
plt.show()

# ヒストグラムの作成
# --- 基準化データ
pd.Series(preprocessing.scale(X[:, 2])).hist(bins=50)
plt.show()


# 3 クラスを用いた基準化 -------------------------------------------------------------------

# インスタンス生成
my_scaler = preprocessing.StandardScaler()

# 基準化
my_scaler.fit(X[:, :3])

# 出力
my_scaler.mean_
my_scaler.var_


# 平均値の出力
my_scaler.transform(X[:, :3]).mean(axis=0)


# 4 Max-Min Scaling -----------------------------------------------------------------------

# インスタンス生成
my_minmax_scaler = preprocessing.MinMaxScaler()

# スケーリング
my_minmax_scaler.fit(X[:, :3])

# レンジ確認
my_minmax_scaler.transform(X[:, :3]).max(axis=0)
my_minmax_scaler.transform(X[:, :3]).min(axis=0)

# インスタンス生成
# --- スケール範囲の指定
my_odd_scaler = preprocessing.MinMaxScaler(feature_range=(-3.14, 3.14))
my_odd_scaler.fit(X[:, :3])
my_odd_scaler.transform(X[:, :3]).max(axis=0)
my_odd_scaler.transform(X[:, :3]).min(axis=0)

#
normalized_X = preprocessing.normalize(X[:, :3])
normalized_X.sum(axis=1)
