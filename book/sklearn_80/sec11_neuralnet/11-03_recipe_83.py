# ******************************************************************************
# Chapter   : 11 ニューラルネットワーク
# Title     : 11-3 多層パーセプトロン（Recipe83)
# Created by: Owner
# Created on: 2020/12/30
# Page      : P324 - P328
# ******************************************************************************

# ＜概要＞
#


# ＜目次＞
# 0 準備
# 1 前処理
# 2 モデリング


# 0 準備 ------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor


# データロード
cali_housing = fetch_california_housing()

# データ格納
X = cali_housing.data
y = cali_housing.target

# 連続データの離散化
bins = np.arange(6)
binned_y = np.digitize(y, bins)

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=binned_y)


# 1 前処理 -----------------------------------------------------------------------------------

# インスタンス生成
scaler = StandardScaler()

# 変換器の作成
scaler.fit(X_train)

# データ変換
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 2 モデリング ------------------------------------------------------------------------------

# パラメータ設定
param_grid = {
    'alpha': [10, 1, 0.1, 0.01],
    'hidden_layer_sizes': [(50, 50, 50), (50, 50, 50, 50, 50)],
    'activation': ['relu', 'logistic'],
    'solver': ['adam']
}

# インスタンス生成
pre_gs_inst = RandomizedSearchCV(MLPRegressor(random_state=7),
                                 param_distributions=param_grid,
                                 cv=3, n_iter=15, random_state=7)

# 学習
pre_gs_inst.fit(X_train_scaled, y_train)

# 最良スコア
pre_gs_inst.best_score_
