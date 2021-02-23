# ******************************************************************************
# Chapter   : 8 サポートベクトルマシン
# Title     : 8-5 サポートベクトル回帰（Recipe68)
# Created by: Owner
# Created on: 2020/12/30
# Page      : P255 - P256
# ******************************************************************************

# ＜概要＞
# -


# ＜目次＞
# 0 準備
# 1 パイプラインの構築
# 2 チューニング


# 0 準備 -----------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit


# データロード
diabetes = datasets.load_diabetes()

# データ格納
X = diabetes.data
y = diabetes.target

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)


# 1 パイプラインの構築 -----------------------------------------------------------------------

# パイプラインの定義
svm_est = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', OneVsRestClassifier(SVR()))
])

# パラメータ設定
Cs = [0.001, 0.01, 0.1, 1]
gammas = [0.001, 0.01, 0.1]
param_grid = dict(svc__estimator__gamma=gammas, svc__estimator__C=Cs)


# 2 チューニング ---------------------------------------------------------------------------

# インスタンス生成
rand_grid = RandomizedSearchCV(svm_est,
                               param_distributions=param_grid,
                               cv=5, n_iter=5, scoring='neg_mean_absolute_error')

# チューニング実行
rand_grid.fit(X_train, y_train)

# 最良パラメータ
rand_grid.best_params_

# ベストスコア
rand_grid.best_score_
