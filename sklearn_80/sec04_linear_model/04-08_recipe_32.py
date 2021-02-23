# ******************************************************************************
# Chapter   : 4 線形モデル - 線形回帰からLARSまで
# Title     : 4-8 LARSによる正則化へのより基本的なアプローチ（Recipe32)
# Created by: Owner
# Created on: 2020/12/25
# Page      : P126 - P129
# ******************************************************************************

# ＜概要＞
# -


# ＜目次＞
# 0 準備
# 1 LARSによる学習
# 2 LARSモデルの実行比較
# 3 特徴量選択としてのLARS


# 0 準備 ------------------------------------------------------------------------------------------

import numpy as np

from sklearn.datasets import make_regression
from sklearn.linear_model import Lars
from sklearn.linear_model import  LarsCV


# データロード
reg_data, reg_target = make_regression(n_samples=200, n_features=500,
                                       n_informative=10, noise=2)


# 1 LARSによる学習 --------------------------------------------------------------------------

# インスタンス生成
lars = Lars(n_nonzero_coefs=10)

# 学習
lars.fit(reg_data, reg_target)

# 比ゼロ係数の数
np.sum(lars.coef_ != 0)


# 2 LARSモデルの実行比較 ---------------------------------------------------------------------

# ＜ポイント＞
# - データを半分に取り分けてLARSモデルで学習


# 変数定義
# --- 訓練データ数
train_n = 100

# インスタンス生成と学習
# --- 非ゼロ係数の数を12個とする
lars_12 = Lars(n_nonzero_coefs=12)
lars_12.fit(reg_data[:train_n], reg_target[:train_n])

# インスタンス生成と学習
# --- 非ゼロ係数の数を500個とする（デフォルト）
lars_500 = Lars(n_nonzero_coefs=500)
lars_500.fit(reg_data[:train_n], reg_target[:train_n])

# 平均二乗誤差
np.mean(np.power(
    reg_target[train_n:] - lars_500.predict(reg_data[train_n:]), 2))


# 3 特徴量選択としてのLARS ---------------------------------------------------------------------

# インスタンス生成
lcv = LarsCV()

# 学習
lcv.fit(reg_data, reg_target)

# 非ゼロの係数
np.sum(lcv.coef_ != 0)
