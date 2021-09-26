# ******************************************************************************
# Book Name : 機械学習を解釈する技術
# Chapter   : インスタンスごとの特異性をとらえる
# Theme     : 5-2 交互作用とPDの限界
# Created on: 2021/09/25
# Page      : P134 - P140
# URL       : https://github.com/ghmagazine/ml_interpret_book
# ******************************************************************************


# ＜概要＞
# - PDによる特徴量と予測値の解釈に限界があることを確認する


# ＜目次＞
# 0 準備
# 1 シミュレーションデータの生成
# 2 ランダムフォレストのモデル構築
# 3 PDによる可視化


# 0 準備 --------------------------------------------------------------------

import sys
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 自作モジュール
sys.path.append("book/ml_interpret_book")
from module.chap5.func import plot_scatter
from mli.metrics import regression_metrics
from mli.interpret import PartialDependence

# その他の設定
# --- ランダムシードの設定
np.random.seed(42)


# 1 シミュレーションデータの生成 -----------------------------------------------

# ＜ポイント＞
# - 交互効果のある特徴量を含むデータセットの作成
# - データ生成過程に入っている二項分布(0/1)で傾きが変わってクロス型のデータが生成される

# 関数定義
def generate_simulation_data():

    # パラメータ設定
    # --- インスタンス数
    N = 1000

    # 特徴量の生成
    # --- X0とX1は一様分布から生成
    # --- X2は二項分布から作成
    x0 = np.random.uniform(low=-1, high=1, size=N)
    x1 = np.random.uniform(low=-1, high=1, size=N)
    x2 = np.random.binomial(n=1, p=0.5, size=N)

    # ノイズは正規分布から生成
    epsilon = np.random.normal(loc=0, scale=0.1, size=N)

    # 特徴量をまとめる
    X = np.column_stack([x0, x1, x2])

    # 線形和で目的変数を作成
    y = x0 - 5 * x1 + 10 * x1 * x2 + epsilon

    return train_test_split(X, y, test_size=0.2, random_state=42)


# シミュレーションデータの生成
X_train, X_test, y_train, y_test = generate_simulation_data()

# プロット作成(X1)
# --- クロスの形をしている（X1はyに対して確実に影響があることを確認）
plot_scatter(X_train[:, 1], y_train, title="Scatter", xlabel="X1", ylabel="y")


# 2 ランダムフォレストのモデル構築 ---------------------------------------------

# ＜ポイント＞
# - ランダムフォレストでは高い精度で予測が可能
#   --- 交互作用も含めて高い精度で学習することができている

# モデル構築
rf = RandomForestRegressor(n_jobs=-1, random_state=42)
rf.fit(X=X_train, y=y_train)

# 予測精度の確認
# --- RMSE:0.31 / R2:0.99
regression_metrics(estimator=rf, X=X_test, y=y_test)


# 3 PDによる可視化 ----------------------------------------------------------

# ＜ポイント＞
# - PDPは計算過程で平均化するため、今回のようなX1のyとの関係性をうまく表現できない
#   --- X1はクロス型になっているが平均するとゼロになってしまうため

# PDの算出
pdp = PartialDependence(estimator=rf, X=X_test, var_names=["X0", "X1", "X2"])

# プロット作成(X1)
# --- X1とyの関係をPDPを作成すると無関係であることが示唆される（散布図と矛盾）
# --- PDがX1とyの関係をうまく表現できていない
pdp.partial_dependence(var_name="X1")
pdp.plot(ylim=(-6, 6))
