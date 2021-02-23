# ******************************************************************************
# Chapter   : 9 決定木アルゴリズムとアンサンブル学習
# Title     : 9-5 回帰に決定木を利用する（Recipe72)
# Created by: Owner
# Created on: 2020/12/29
# Page      : P272 - P277
# ******************************************************************************

# ＜概要＞
# - 決定木を回帰モードで使用する


# ＜目次＞
# 0 準備
# 1 ラベルの離散化
# 2 モデリング
# 3 誤差の分析


# 0 準備 ------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


# データロード
diabetes = load_diabetes()

# データ格納
X = diabetes.data
y = diabetes.target

# 特徴量のラベル
X_feature_names = ['age', 'gender', 'body mass index', 'average blood pressure',
                   'bl_0', 'bl_1', 'bl_2', 'bl_3', 'bl_4', 'bl_5']

# プロット作成
pd.Series(y).hist(bins=50)
plt.rcParams["font.size"] = 10
plt.show()


# 1 ラベルの離散化 --------------------------------------------------------------------------

# ＜ポイント＞
# - 回帰では層化サンプリング(stratify=y)を使用することができない
#   --- 代わりに目的変数を離散化する

# ビンの階級
bins = 50 * np.arange(8)
bins

# 目的変数の離散化
binned_y = np.digitize(y, bins)

# プロット表示
pd.Series(binned_y).hist(bins=50)
plt.show()

# データ分割
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, stratify=binned_y)


# 2 モデリング --------------------------------------------------------------------------

# インスタンス生成
dtr = DecisionTreeRegressor()
vars(dtr)

# 学習
dtr.fit(X_train, y_train)
vars(dtr)

# 予測
y_pred = dtr.predict(X_test)

# モデル評価
mean_absolute_error(y_true=y_test, y_pred=y_pred)

# MAPE
(np.abs(y_test - y_pred) / y_test).mean()


# 3 誤差の分析 --------------------------------------------------------------------------

# ヒストグラム
# --- 誤差
pd.Series(y_test - y_pred).hist(bins=50)
plt.show()

# ヒストグラム
# --- 誤差率
pd.Series((y_test - y_pred) / y_test).hist(bins=50)
plt.show()
