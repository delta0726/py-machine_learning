# ******************************************************************************
# Chapter   : 2 モデル構築前のワークフローと前処理
# Title     : 2-10 回帰に確率的勾配降下法を使用する（Recipe18)
# Created by: Owner
# Created on: 2020/12/25
# Page      : P77 - P80
# ******************************************************************************

# ＜概要＞
# - 確率的勾配降下法(SGD)は多くのアルゴリズムの内部で使用されている
#   --- 単純かつ高速に動作


# ＜目次＞
# 0 準備
# 1 SGDRegressorモデルの適合


# 0 準備 ------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor


# データ準備
# --- 100万行の大規模データセット
X, y = make_regression(int(1e6))

# 行列サイズ
X.shape

# データサイズ
# --- バイト
# --- メガバイト
print("{:,}".format(X.nbytes))
X.nbytes / 1e6

# 参考：データ点あたりのバイト数
X.nbytes / (X.shape[0] * X.shape[1])


# 1 SGDRegressorモデルの適合 -----------------------------------------------------------------------

# インスタンス生成
sgd = SGDRegressor(max_iter=5)

# 乱数ベクトル生成
# --- True/False
# --- データ分割用
train = np.random.choice([True, False], size=len(y), p=[0.25, 0.75])

# 学習
sgd.fit(X[train], y[train])

# 予測
y_pred = sgd.predict(X[~train])
y_pred

# プロット作成
pd.Series(y[~train] - y_pred).hist(bins=50)
plt.show()
