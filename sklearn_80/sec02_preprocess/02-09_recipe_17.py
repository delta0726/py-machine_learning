# ******************************************************************************
# Chapter   : 2 モデル構築前のワークフローと前処理
# Title     : 2-9 回帰にガウス過程を使用する（Recipe17)
# Created by: Owner
# Created on: 2020/12/24
# Page      : P69 - P77
# ******************************************************************************

# ＜概要＞
# - ガウス過程とは平均ではなく分散(バリアンス)に関するプロセス


# ＜目次＞
# 0 準備
# 1 ガウス過程の仕組み
# 2 結果のプロット
# 3 プロットの関数化
# 4 ノイズパラメータを使った交差検証
# 5 推定値の不確実性


# 0 準備 ------------------------------------------------------------------------------------------

import numpy as np

from sklearn.datasets import load_boston
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as CK
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt


# データ準備
boston = load_boston()

# データ格納
boston_X = boston.data
boston_y = boston.target

# データ分割
train_set = np.random.choice([True, False], len(boston_y), p=[0.75, 0.25])


# 1 ガウス過程の仕組み ------------------------------------------------------------------------------

# ＜ハイパーパラメータ＞
# - アルファ
#   --- ノイズパラメータ
#   --- 全ての観測値にノイズ値を割り当てるか、Numpy配列形式でn個の値を割り当てる
#   --- nは訓練データの目的変数の値の長さ
# - kernel
#   --- 関数を近似するカーネル
#   --- ここでは定数カーネルとRBFカーネルから柔軟なカーネルを構築
# - normalize_y
#   --- ターゲットとなるデータセットの平均が0出ない場合はTrueに設定することができる
#   --- Falseのままにしておいても十分にうまくいく
# - n_restarts_optimizer
#   --- カーネルを最適化するためのイテレーション回数
#   --- 実際に使用する場合は10-20に設定


# カーネル関数の設定
mixed_kernel = kernel = CK(1, (1e-4, 1e4)) * RBF(10, (1e-4, 1e4))

# インスタンスの生成
gpr = GaussianProcessRegressor(alpha=5, n_restarts_optimizer=20, kernel=mixed_kernel)
gpr

# 学習
gpr.fit(boston_X[train_set], boston_y[train_set])

# 予測
test_preds = gpr.predict(boston_X[~train_set])


# 2 結果のプロット ---------------------------------------------------------------------------------

# レイアウト設定
f, ax = plt.subplots(figsize=(10, 7), nrows=3)
f.tight_layout()

# プロット1
ax[0].plot(range(len(test_preds)), test_preds, label='Predicted Values')
ax[0].plot(range(len(test_preds)), boston_y[~train_set], label='Actual Values')
ax[0].set_title("Predicted vs Actuals")
ax[0].legend(loc='best')

# プロット2
ax[1].plot(range(len(test_preds)), test_preds - boston_y[~train_set])
ax[1].set_title("Plotted Residuals")

# プロット3
ax[2].hist(test_preds - boston_y[~train_set])
ax[2].set_title("Histogram of Residuals")

# プロット表示
plt.show()


# 3 プロットの関数化 ---------------------------------------------------------------------------------

def plot_result(test_preds):
    # レイアウト設定
    f, ax = plt.subplots(figsize=(10, 7), nrows=3)
    f.tight_layout()

    # プロット1
    ax[0].plot(range(len(test_preds)), test_preds, label='Predicted Values')
    ax[0].plot(range(len(test_preds)), boston_y[~train_set], label='Actual Values')
    ax[0].set_title("Predicted vs Actuals")
    ax[0].legend(loc='best')

    # プロット2
    ax[1].plot(range(len(test_preds)), test_preds - boston_y[~train_set])
    ax[1].set_title("Plotted Residuals")

    # プロット3
    ax[2].hist(test_preds - boston_y[~train_set])
    ax[2].set_title("Histogram of Residuals")

    # プロット表示
    plt.show()


# 4 ノイズパラメータを使った交差検証 -------------------------------------------------------------------

# 関数定義
def score_mini_report(score_list):
    print("List of scores", score_list)
    print("Mean of scores", score_list.mean())
    print("Std of scores", score_list.std())


# ケース1： alpha=5 ***********************************************************************

# インスタンス生成
gpr5 = GaussianProcessRegressor(alpha=5,
                                n_restarts_optimizer=20,
                                kernel=mixed_kernel)

# インスタンス生成
# --- クロスバリデーション
scores_5 = (cross_val_score(gpr5,
                            boston_X[train_set], boston_y[train_set],
                            cv=4,
                            scoring='neg_mean_absolute_error'))


# 検証結果
score_mini_report(scores_5)


# ケース2： alpha=7 ***********************************************************************

# インスタンス生成
gpr7 = GaussianProcessRegressor(alpha=7,
                                n_restarts_optimizer=20,
                                kernel=mixed_kernel)


# インスタンス生成
scores_7 = (cross_val_score(gpr7,
                            boston_X[train_set], boston_y[train_set],
                            cv=4,
                            scoring='neg_mean_absolute_error'))

# 検証結果
score_mini_report(scores_7)


# ケース3： alpha=7 & normalize=True *********************************************************

# インスタンス生成
gpr7n = GaussianProcessRegressor(alpha=7,
                                 n_restarts_optimizer=20,
                                 kernel=mixed_kernel,
                                 normalize_y=True)


# インスタンス生成
scores_7n = (cross_val_score(gpr7n,
                             boston_X[train_set], boston_y[train_set],
                             cv=4,
                             scoring='neg_mean_absolute_error'))

# 検証結果
score_mini_report(scores_7n)

# 学習
gpr7n.fit(boston_X[train_set], boston_y[train_set])

# 予測
test_preds = gpr7n.predict(boston_X[~train_set])
test_preds

# プロット確認
plot_result(test_preds)


# ケース4： alphaを調整 *********************************************************

# インスタンス生成
gpr_new = GaussianProcessRegressor(alpha=boston_y[train_set]/4,
                                   n_restarts_optimizer=20,
                                   kernel=mixed_kernel)

# 学習
gpr_new.fit(boston_X[train_set], boston_y[train_set])

# 予測
test_preds = gpr_new.predict(boston_X[~train_set])

# プロット確認
plot_result(test_preds)


# 5 推定値の不確実性 --------------------------------------------------------------------------

# MSEの推定値
test_preds, MSE = gpr7n.predict(boston_X[~train_set], return_std=True)
MSE[:5]


# 関数定義
def plot_uncertainty(n):
    f, ax = plt.subplots(figsize=(7, 5))
    rng = range(n)
    ax.scatter(rng, test_preds[:n])
    ax.errorbar(rng, test_preds[:n], yerr=1.96*MSE[:n])
    ax.set_title("Predictions with Error Bars")
    ax.set_xlim((-1, n))
    plt.show()


# プロット作成
plot_uncertainty(n=MSE.shape[0])

# プロット作成
plot_uncertainty(n=20)
