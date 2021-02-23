# ******************************************************************************
# Chapter   : 4 線形モデル - 線形回帰からLARSまで
# Title     : 4-3 機械学習を使って直線をデータに適合させる（Recipe27)
# Created by: Owner
# Created on: 2020/12/25
# Page      : P108 - P109
# ******************************************************************************

# ＜概要＞
# - 機械学習における線形回帰は未知のデータの予測に使うことを想定している
#   --- クロスバリデーションでモデルを評価/予測するのが理にかなっている
#   --- ただし、以下の例では訓練データ/テストデータの分割は行っていなし


# ＜目次＞
# 0 準備
# 1 線形回帰の実行


# 0 準備 ------------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict


# データロード
boston = load_boston()


# 1 線形回帰の実行 -------------------------------------------------------------------------------

# インスタンス生成
lr = LinearRegression()
vars(lr)

# 学習
lr.fit(boston.data, boston.target)
vars(lr)

# 予測
# --- クロスバリデーションで予測値を作成
predictions_cv = cross_val_predict(lr, boston.data, boston.target, cv=10)
predictions_cv[:5]

# プロット作成
# --- 交差検証なしの線形回帰よりも左右対称的
pd.Series(boston.target - predictions_cv).hist(bins=50)
plt.show()


