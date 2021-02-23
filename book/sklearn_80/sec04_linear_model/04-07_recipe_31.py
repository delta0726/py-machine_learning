# ******************************************************************************
# Chapter   : 4 線形モデル - 線形回帰からLARSまで
# Title     : 4-7 疎性を使ってモデルを正則化する（Recipe31)
# Created by: Owner
# Created on: 2020/12/25
# Page      : P123 - P126
# ******************************************************************************

# ＜概要＞
# - LASSOはリッジ回帰やLARS(Least Angle Regression)によく似た手法
#   --- リッジ回帰との類似点：ペナルティを課すこと
#   --- LARSとの類似点：パラメータ選択に利用できること


# ＜目次＞
# 0 準備
# 1 LASSO回帰の実行
# 2 ラムダの強さの選択


# 0 準備 ------------------------------------------------------------------------------------------

import numpy as np

from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV


# データロード
reg_data, reg_target = make_regression(n_samples=200, n_features=500,
                                       n_informative=5, noise=5)

# データ確認
# --- 特徴量(X)は500個
reg_data.shape
reg_target.shape


# 1 LASSO回帰の実行 -------------------------------------------------------------------------------

# インスタンス生成
# --- alphaはデフォルトの1のまま
lasso = Lasso()
vars(lasso)

# 学習
lasso.fit(reg_data, reg_target)
vars(lasso)

# 0でない係数
np.sum(lasso.coef_ != 0)

# alphaを0に変更
# --- 変数選択なし（Ridge回帰）
lasso_0 = Lasso(0)
lasso_0.fit(reg_data, reg_target)
np.sum(lasso_0.coef_ != 0)


# 2 ラムダの強さの選択 --------------------------------------------------------------------------

# ＜ポイント＞
# - 最適なラムダ(L2正則化ペナルティの強さ)の選択は重要な問題

# インスタンス生成
lassocv = LassoCV()
vars(lassocv)

# 学習
lassocv.fit(reg_data, reg_target)
vars(lassocv)

# アルファの確認
lassocv.alpha

# 係数
lassocv.coef_[:5]

# 0以外の回帰係数
np.sum(lassocv.coef_ != 0)


# 3 特徴量選択に対するLASSO --------------------------------------------------------------------

# 回帰係数
# --- 0以外をTRUE/0をFALSE
mask = lassocv.coef_ != 0

# データセットの選択
# --- 回帰係数0の特徴量を除外
new_reg_data = reg_data[:, mask]

# 特徴量選択後のデータセット
new_reg_data.shape
