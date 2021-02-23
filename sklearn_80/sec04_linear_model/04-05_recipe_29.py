# ******************************************************************************
# Chapter   : 4 線形モデル - 線形回帰からLARSまで
# Title     : 4-5 リッジ回帰を使って線形回帰の欠点を克服する（Recipe29)
# Created by: Owner
# Created on: 2020/12/25
# Page      : P114 - P118
# ******************************************************************************

# ＜概要＞
# - リッジ回帰は回帰係数を収縮するための正則化パラメータを持つ
#   --- 多重共線性の問題は自動的に解消する


# ＜目次＞
# 0 準備
# 1 線形回帰における回帰係数の安定性
# 2 リッジ回帰における回帰係数の安定性


# 0 準備 ------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge


# データロード
boston = load_boston()


# データセット生成
# --- 有効階数はその行列が厳密にはフルランクであるものの、列の多くが共線性を持つことを意味する
reg_data, reg_target = make_regression(n_samples=2000, n_features=3,
                                       effective_rank=2, noise=10)


# 1 線形回帰における回帰係数の安定性 ---------------------------------------------------------------

# ＜ポイント＞
# - リッジ回帰と比べると変数の安定性は低い


# インスタンス生成
# --- 線形回帰
lr = LinearRegression()

# 変数定義
n_bootstrap = 1000
len_data = len(reg_data)
subsample_size = np.int(0.5 * len_data)

# サンプリング
subsample = lambda: np.random.choice(np.arange(0, len_data),
                                     size=subsample_size)

# オブジェクト作成
# --- coefの領域確保
coefs_lr = np.ones((n_bootstrap, 3))

# シミュレーション
# --- ブートストラップサンプリングを用いた回帰係数の安定性検証
# --- 50％の復元抽出を1000回
for i in range(n_bootstrap):
    subsample_idx = subsample()
    subsample_X = reg_data[subsample_idx]
    subsample_y = reg_target[subsample_idx]
    lr.fit(subsample_X, subsample_y)
    coefs_lr[i][0] = lr.coef_[0]
    coefs_lr[i][1] = lr.coef_[1]
    coefs_lr[i][2] = lr.coef_[2]


# 関数定義
# --- プロット作成
def plot_result(coef):
    plt.figure(figsize=(10, 5))

    # 回帰係数1
    ax1 = plt.subplot(311, title='Coef 0')
    ax1.hist(coef[:, 0])

    # 回帰係数2
    ax2 = plt.subplot(312, sharex=ax1, title='Coef 1')
    ax2.hist(coef[:, 1])

    # 回帰係数3
    ax3 = plt.subplot(313, sharex=ax1, title='Coef 2')
    ax3.hist(coef[:, 2])

    # 出力設定
    plt.tight_layout()
    plt.show()


# プロット出力
plot_result(coef=coefs_lr)


# 2 リッジ回帰における回帰係数の安定性 ------------------------------------------------------------

# ＜ポイント＞
# - 線形回帰と比べると変数の安定性は高い


# インスタンス生成
# --- リッジ回帰
ridge = Ridge()

# 変数定義
n_bootstrap = 1000
len_data = len(reg_data)
subsample_size = np.int(0.5 * len_data)

# サンプリング
subsample = lambda: np.random.choice(np.arange(0, len_data),
                                     size=subsample_size)

# オブジェクト作成
# --- coefの領域確保
coefs_ridge = np.ones((n_bootstrap, 3))

# シミュレーション
# --- ブートストラップサンプリングを用いた回帰係数の安定性検証
# --- 50％の復元抽出を1000回
for i in range(n_bootstrap):
    subsample_idx = subsample()
    subsample_X = reg_data[subsample_idx]
    subsample_y = reg_target[subsample_idx]
    ridge.fit(subsample_X, subsample_y)
    coefs_ridge[i][0] = ridge.coef_[0]
    coefs_ridge[i][1] = ridge.coef_[1]
    coefs_ridge[i][2] = ridge.coef_[2]


# プロット出力
plot_result(coef=coefs_ridge)


# 3 回帰係数の分散比較 ------------------------------------------------------------

# 線形回帰
np.var(coefs_lr, axis=0)

# リッジ回帰
np.var(coefs_ridge, axis=0)
