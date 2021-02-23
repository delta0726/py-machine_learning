# ******************************************************************************
# Chapter   : 4 線形モデル - 線形回帰からLARSまで
# Title     : 4-6 リッジ回帰のパラメータを最適化する（Recipe30)
# Created by: Owner
# Created on: 2020/12/25
# Page      : P118 - P123
# ******************************************************************************

# ＜概要＞
# - リッジ回帰ではアルファパラメータを決定する必要がある


# ＜目次＞
# 0 準備
# 1 アルファの調整
# 2 最良アルファの考え方
# 3 独自スコアを用いた評価


# 0 準備 ------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import RidgeCV
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer


# データロード
reg_data, reg_target = make_regression(n_samples=100, n_features=2,
                                       effective_rank=1, noise=10)

# データ確認
reg_data.shape
reg_target.shape


# 1 アルファの調整 ----------------------------------------------------------------------------

# ＜ポイント＞
# - RidgeCVは1個抜き交差検証と同様の交差検証を行う

# インスタンス生成
# --- アルファを複数持たせてチューニング
rcv = RidgeCV(alphas=np.array([0.1, 0.2, 0.3, 0.4]))
vars(rcv)

# 学習
rcv.fit(reg_data, reg_target)
vars(rcv)

# 最適アルファ
rcv.alpha_

# 最適アルファ
# --- もう少し細かく調整
rcv = RidgeCV(alphas=np.array([0.08, 0.09, 0.10, 0.11, 0.12]))
rcv.fit(reg_data, reg_target)
rcv.alpha_


# 2 最良アルファの考え方 ---------------------------------------------------------------------

# アルファのテスト値
# --- 50個を線形で作成
alpha_to_test = np.linspace(0.01, 1)

# クロスバリデーション
# --- 結果をオブジェクトに格納
rcv3 = RidgeCV(alphas=alpha_to_test, store_cv_values=True)
vars(rcv3)

# 学習
rcv3.fit(reg_data, reg_target)
vars(rcv3)

# アルファの結果
# --- 50個
rcv3.cv_values_.shape

# 平均誤差が最小となるアルファ
# --- CVの誤差が最小となるインデックスのアルファを取得
smallest_idx = rcv3.cv_values_.mean(axis=0).argmin()
alpha_to_test[smallest_idx]

# CVの出力値
rcv3.alpha_

# プロット作成
# --- 上記の最小値とイメージ一致
# --- 乱数シードを固定していないため若干結果が異なる
plt.plot(alpha_to_test, rcv3.cv_values_.mean(axis=0))
plt.show()


# 3 独自スコアを用いた評価 ------------------------------------------------------------------

# スコア定義
# --- 関数からクラスを生成
MAD_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
type(mean_absolute_error)
type(MAD_scorer)

# クロスバリデーション
# --- 結果をオブジェクトに格納
rcv4 = RidgeCV(alphas=alpha_to_test, store_cv_values=True, scoring=MAD_scorer)
vars(rcv4)

# 学習
rcv4.fit(reg_data, reg_target)
vars(rcv4)

# 平均誤差が最小となるアルファ
smallest_idx = rcv4.cv_values_.mean(axis=0).argmin()
rcv4.cv_values_.mean(axis=0)[smallest_idx]
alpha_to_test[smallest_idx]

# CVの出力値
rcv4.alpha_
