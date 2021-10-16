# ******************************************************************************
# Book Name : 機械学習を解釈する技術
# Chapter   : インスタンスごとの特異性をとらえる
# Theme     : 5-4 Conditional Partial Dependence
# Created on: 2021/09/26
# Page      : P152 - P159
# URL       : https://github.com/ghmagazine/ml_interpret_book
# ******************************************************************************


# ＜概要＞
# - ICEは各インスタンスに対して個別に特徴量と予測値のwhat-ifの解釈をすることができる
# - ICEの解釈においては依存関係のある特徴量を含むモデルでは注意を要する


# ＜目次＞
# 0 準備
# 1 what-if
# 2 シミュレーションデータの生成（特徴量に依存関係がある場合）


# 0 準備 ----------------------------------------------------------------------

import sys

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from module.chap5.ice import IndividualConditionalException
from module.chap5.data import generate_simulation_data
from module.chap5.func import plot_scatter
from mli.metrics import regression_metrics


# シミュレーションデータの生成
X_train, X_test, y_train, y_test = generate_simulation_data()

# モデル構築
rf = RandomForestRegressor(n_jobs=-1, random_state=42)
rf.fit(X=X_train, y=y_train)


# 1 what-if -----------------------------------------------------------------

# ＜ポイント＞
# - ICEは各インスタンスに対して個別にwhat-ifの解釈を与えている

# インスタンスの生成
ice = IndividualConditionalException(estimator=rf, X=X_test, var_names=["X0", "X1", "X2"])


# ICEの計算
# --- 特徴量：X1 / インスタンス： 0
ice.individual_conditional_exception(var_name="X1", ids_to_compute=[0, 1])

# ICEのプロット
ice.plot(ylim=(-6, 6))

# 特徴量の値
ice.df_instance


# 2 シミュレーションデータの生成（特徴量に依存関係がある場合） ----------------------

# ＜ポイント＞
# - 依存関係のある特徴量を含むデータセットの生成


def generate_simulation_data():
    # パラメータ設定
    # --- インスタンス数
    N = 1000

    # 特徴量の定義
    # --- X0：一様分布から生成
    # --- X2：二項分布から生成（試行回数1にすると確率を0.5のベルヌーイ分布に一致）
    # --- X1：X1はX2に依存する形にする
    x0 = np.random.uniform(-1, 1, N)
    x2 = np.random.binomial(1, 0.5, N)
    x1 = np.where(x2 == 1,
                  np.random.uniform(-0.5, 1, N),
                  np.random.uniform(-1, 0.5, N))

    # ノイズは正規分布から生成
    epsilon = np.random.normal(loc=0, scale=0.1, size=N)

    # 特徴量をまとめる
    X = np.column_stack((x0, x1, x2))

    # 線形和で目的変数を作成
    y = x0 - 5 * x1 + 10 * x1 * x2 + epsilon

    return train_test_split(X, y, test_size=0.2, random_state=42)


# データ出力
X_train, X_test, y_train, y_test = generate_simulation_data()

# プロット作成
plot_scatter(X=X_train[:, 1], y=y_train, group=X_train[:, 2],
             xlabel="X1", ylabel="y", title="Scatter Plot")


# 3 依存関係のある特徴量の解釈 -------------------------------------------------

# モデル構築
rf = RandomForestRegressor(n_jobs=-1, random_state=42)
rf.fit(X=X_train, y=y_train)

# 予測精度の確認
regression_metrics(estimator=rf, X=X_test, y=y_test)

# ICEの計算
# --- 特徴量：X1 / インスタンス：0
ice = IndividualConditionalException(estimator=rf, X=X_test, var_names=["X0", "X1", "X2"])
ice.individual_conditional_exception(var_name="X1", ids_to_compute=[0])

# インスタンスの特徴量を確認
ice.df_instance

# ICEの可視化
ice.plot(ylim=(-6, 6))
