# ******************************************************************************
# Chapter   : 9 決定木アルゴリズムとアンサンブル学習
# Title     : 9-11 scikit-learnでスタッキングアグリゲータを作成する（Recipe78)
# Created by: Owner
# Created on: 2020/12/30
# Page      : P298 - P305
# ******************************************************************************

# ＜概要＞
# -


# ＜目次＞
# 0 準備


# 0 準備 ------------------------------------------------------------------------------------------

import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import AdaBoostRegressor


# データロード
cali_housing = fetch_california_housing()

# データ格納
X = cali_housing.data
y = cali_housing.target

