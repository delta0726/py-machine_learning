# ******************************************************************************
# Chapter   : 4 線形モデル - 線形回帰からLARSまで
# Title     : 4-4 線形回帰モデルを評価する（Recipe28)
# Created by: Owner
# Created on: 2020/12/25
# Page      : P109 - P114
# ******************************************************************************

# ＜概要＞
# - 予測結果がどのくらいうまく適合しているのかを定量化する


# ＜目次＞
# 0 準備
# 1 モデル構築
# 2 QQプロットの作成
# 3 評価指標の定義と出力
# 4 評価指標の関数による出力
# 5 回帰係数の安定性


# 0 準備 ------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error

from scipy.stats import probplot


# データロード
boston = load_boston()


# 1 モデル構築 ---------------------------------------------------------------------------------

# インスタンス生成
lr = LinearRegression()

# 学習
lr.fit(boston.data, boston.target)

# 予測
predictions_cv = cross_val_predict(lr, boston.data, boston.target, cv=10)
predictions_cv[:10]


# 2 QQプロットの作成 --------------------------------------------------------------------------

# プロット設定
f = plt.figure(figsize=(7, 5))
ax = f.add_subplot(111)

# QQプロットの作成
# --- 予測値と観測値から予測の正規性を確認
tuple_out = probplot(boston.target - predictions_cv, plot=ax)
plt.show()

# 出力
# --- (slope, intercept, rsq)が出力される
tuple_out[1]


# 3 評価指標の定義と出力 ------------------------------------------------------------------------

# 関数定義
# --- 平均二乗誤差(MSE)
def MSE(target, predictions):
    squared_deviation = np.power(target - predictions, 2)
    return np.mean(squared_deviation)


# 関数定義
# --- 平均絶対偏差(MAD)
def MAD(target, predictions):
    absolute_deviation = np.abs(target - predictions)
    return np.mean(absolute_deviation)


# 出力
MSE(boston.target, predictions_cv)
MAD(boston.target, predictions_cv)


# 4 評価指標の関数による出力 -------------------------------------------------------------------

# 出力
print('MSE', mean_squared_error(boston.target, predictions_cv))
print('MAE', mean_absolute_error(boston.target, predictions_cv))


# 5 回帰係数の安定性 --------------------------------------------------------------------------

# 変数定義
n_bootstrap = 1000
len_boston = len(boston.target)
subsample_size = np.int(0.5 * len_boston)

# サンプリング
subsample = lambda: np.random.choice(np.arange(0, len_boston),
                                     size=subsample_size)

# オブジェクト作成
# --- coefの領域確保
coefs = np.ones(n_bootstrap)

# シミュレーション
# --- ブートストラップサンプリングを用いた回帰係数の安定性検証
# --- 50％の復元抽出を1000回
for i in range(n_bootstrap):
    subsample_idx = subsample()
    subsample_X = boston.data[subsample_idx]
    subsample_y = boston.target[subsample_idx]
    lr.fit(subsample_X, subsample_y)
    coefs[i] = lr.coef_[0]

# プロット作成
f = plt.figure(figsize=(7, 5))
ax = f.add_subplot(111)
ax.hist(coefs, bins=50)
ax.set_title("Histogram of the lr.coef_")
plt.show()

# 信頼区間の取得
np.percentile(coefs, [2.5, 97.5])
