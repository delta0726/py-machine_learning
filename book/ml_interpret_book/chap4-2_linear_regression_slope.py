# ******************************************************************************
# Book Name : 機械学習を解釈する技術
# Chapter   : 特徴量と予測の関係を知る
# Theme     : 4-2 線形回帰モデルと回帰係数
# Created on: 2021/09/21
# Page      : P91 - P98
# URL       : https://github.com/ghmagazine/ml_interpret_book
# ******************************************************************************


# ＜概要＞
# - シミュレーションデータの作成プロセスとプロットを確認する
# - 線形回帰モデルにおける変数重要度を回帰係数で確認する


# ＜目次＞
# 0 準備
# 1 シミュレーションデータの生成（線形）
# 2 線形データによる回帰係数の確認
# 3 シミュレーションデータの生成（非線形）
# 4 非線形データによる回帰係数の確認
# 5 線形回帰モデルによる予測
# 6 ランダムフォレストによる予測


# 0 準備 -----------------------------------------------------------------------

# ライブラリ
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 自作モジュール
sys.path.append("book/ml_interpret_book")
from module.chap4.func import plot_scatter
from mli.metrics import regression_metrics
from mli.utility import get_coef

# その他の設定
# --- ランダムシードの設定
np.random.seed(42)


# 1 シミュレーションデータの生成（線形） --------------------------------------------

# 関数定義
# --- 線形のモデル用データの生成
def generate_simulation_data1():

    # パラメータ設定
    # --- インスタンス数
    # --- 回帰係数
    N = 1000
    beta = np.array([1])

    # 特徴量の生成
    # --- 一様分布を元にする
    X = np.random.uniform(low=0, high=1, size=[N, 1])

    # ノイズの生成
    epsilon = np.random.normal(loc=0, scale=0.1, size=N)

    # 目的変数を作成
    # --- 線形和
    y = X @ beta + epsilon

    # データ分割
    return train_test_split(X, y, test_size=0.2, random_state=42)


# シミュレーションデータの生成
X_train, X_test, y_train, y_test = generate_simulation_data1()

# プロット作成
# --- 散布図
plot_scatter(X=X_train[:, 0], y=y_train, title="Scatter Plot between X and Y")


# 2 線形データによる回帰係数の確認 -------------------------------------------------

# ＜ポイント＞
# - 線形のシミュレーションデータを用いて線形回帰モデルを構築する
#   --- データの実体を反映したモデルが作成されている（予測精度と回帰係数により確認）


# 学習
# --- 線形回帰モデル
lm = LinearRegression().fit(X=X_train, y=y_train)

# 予測精度の確認
# --- RMSE = 0.09 / R2 = 0.91
regression_metrics(estimator=lm, X=X_test, y=y_test)

# 回帰係数の確認
# --- intercept = 2 / X = 0.98
df_coef = get_coef(estimator=lm, var_names=["X"])
df_coef


# 3 シミュレーションデータの生成（非線形） -------------------------------------------

def generate_simulation_data2():

    # パラメータ設定
    # --- 回帰係数
    N = 1000

    # 一様分布から特徴量を生成
    X = np.random.uniform(low=-np.pi * 2, high=np.pi * 2, size=[N, 2])
    epsilon = np.random.normal(loc=0, scale=0.1, size=N)

    # yはsin関数で変換する
    y = 10 * np.sin(X[:, 0]) + X[:, 1] + epsilon

    return train_test_split(X, y, test_size=0.2, random_state=42)


# シミュレーションデータの生成
X_train, X_test, y_train, y_test = generate_simulation_data2()


# 4 非線形データによる回帰係数の確認 -----------------------------------------------

def plot_scatters(X, y, var_name, title=None):

    # 特徴量の数だけ散布図を作成
    J = X.shape[1]
    fig, axes = plt.subplots(nrows=1, ncols=J, figsize=(4 * J, 4))

    # プロット作成
    for j, ax in enumerate(axes):
        sns.scatterplot(X[:, j], y, ci=None, alpha=0.3, ax=ax)
        ax.set(
            xlabel=var_name[j],
            ylabel="Y",
            xlim=(X.min() * 1.1, X.max() * 1.1)
        )
    fig.suptitle(title)

    # プロット出力
    fig.show()


# プロット作成
# --- 特徴量ごとに目的変数との散布図を作成
plot_scatters(X=X_train, y=y_train, var_name=["X0", "X1"],
              title="Scatter plot between features and label")


# 5 線形回帰モデルによる予測 -------------------------------------------------------

# モデル構築
lm = LinearRegression()
lm.fit(X_train, y_train)

# 予測精度の確認
# --- RMSE=6.59 / R2=0.35
regression_metrics(estimator=lm, X=X_test, y=y_test)

# 回帰係数の取得
df_coef = get_coef(lm, ["X0", "X1"])
df_coef


# 6 ランダムフォレストによる予測 ---------------------------------------------------

from sklearn.ensemble import RandomForestRegressor

# モデル構築
rf = RandomForestRegressor()
rf.fit(X=X_train, y=y_train)

# 予測精度の確認
# --- RMSE=0.73 / R2=0.99
regression_metrics(estimator=rf, X=X_test, y=y_test)
