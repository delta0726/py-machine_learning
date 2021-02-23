# ******************************************************************************
# Chapter   : 4 線形モデル - 線形回帰からLARSまで
# Title     : 4-2 直線をデータに適合させる（Recipe26)
# Created by: Owner
# Created on: 2020/12/25
# Page      : P104 - P107
# ******************************************************************************

# ＜概要＞
# -


# ＜目次＞
# 0 準備


# 0 準備 ------------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LinearRegression


# データロード
boston = datasets.load_boston()


# 1 最も単純な線形回帰 ------------------------------------------------------------------------------

# インスタンス生成
lr = LinearRegression()
vars(lr)

# 学習
lr.fit(boston.data, boston.target)
vars(lr)

# 予測
predictions = lr.predict(boston.data)
predictions[:5]

# プロット作成
pd.Series(boston.target - predictions).hist(bins=50)
plt.show()

# 回帰係数
# --- 学習器の中ではラベルを持たない
pd.DataFrame({'Name': boston.feature_names,
              'Coef': lr.coef_})

# 切片項
lr.intercept_


# 2 標準化回帰のパラメータ -----------------------------------------------------------------------

# ＜ポイント＞
# - 線形回帰はデータセットの正規化が必要なケースが多い
#   --- 引数で設定することができる

# インスタンス生成
# --- デフォルトではnormalize引数はFalse
lr2 = LinearRegression(normalize=True)
vars(lr2)

# 予測
lr2.fit(boston.data, boston.target)
vars(lr2)

# 予測
prediction2 = lr2.predict(boston.data)
prediction2[:5]

