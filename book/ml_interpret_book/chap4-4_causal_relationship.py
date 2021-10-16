# ******************************************************************************
# Book Name : 機械学習を解釈する技術
# Chapter   : 特徴量と予測の関係を知る
# Theme     : 4-4 Partial Dependenceは因果関係として解釈できるのか
# Created on: 2021/09/23
# Page      : P113 - P120
# URL       : https://github.com/ghmagazine/ml_interpret_book
# ******************************************************************************


# ＜概要＞
# - Partial Dependence Plotをクラスで実装する


# ＜目次＞
# 0 準備
# 1 シミュレーションデータの作成
# 2 PDによる解釈


# 0 準備 ---------------------------------------------------------------------

# ライブラリ
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 自作モジュール
from mli.metrics import regression_metrics
from module.chap4.func import plot_scatters
from module.chap4.pdp import PartialDependence


# 1 シミュレーションデータの作成 -----------------------------------------------

# ＜ポイント＞
# - X1はyに影響するが、X0はyに影響しないシミュレーションデータを生成する
# - しかし、散布図を作成すると両方がyに対して相関があるように見える
#   --- yとX1に相関があり、X1とX0にも相関があるため、結果としてyとX0に相関が出ている
#   --- 疑似相関であるが、因果関係として誤って解釈してしまう可能性がある

# 関数定義
# --- X1は影響するが、X0は影響しないデータの生成
def generate_simulation_data_3():

    # パラメータ設定
    # --- インスタンス数
    N = 10000

    # 回帰係数
    # --- X1は影響するが、X0は影響しない
    beta = np.array([0, 1])

    # 多変量正規分布から強く相関するデータを生成
    mu = np.array([0, 0])
    Sigma = np.array([[1, 0.95], [0.95, 1]])
    X = np.random.multivariate_normal(mu, Sigma, N)

    # ノイズ生成
    epsilon = np.random.normal(0, 0.1, N)

    # 目的変数の作成
    y = X @ beta + epsilon

    return train_test_split(X, y, test_size=0.2, random_state=42)


# シミュレーションデータの生成
X_train, X_test, y_train, y_test = generate_simulation_data_3()

# プロット作成
plot_scatters(X=X_train, y=y_train, var_names=["X0", "X1"])


# 2 PDによる解釈 --------------------------------------------------------------

# モデル構築
rf = RandomForestRegressor(n_jobs=-1, random_state=42)
rf.fit(X=X_train, y=y_train)

# 予測精度の確認
# --- 予測精度は非常に高い
# --- RMSE: 0.12 / R2: 0.98
regression_metrics(estimator=rf, X=X_test, y=y_test)

# PDの計算
pdp = PartialDependence(estimator=rf, X=X_test, var_names=["X0", "X1"])

# PDによる可視化
pdp.partial_dependence("X0", n_grid=50)
pdp.plot()

# PDによる可視化
# --- X1
pdp.partial_dependence("X1", n_grid=50)
pdp.plot()
