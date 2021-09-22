# ******************************************************************************
# Book Name : 機械学習を解釈する技術
# Chapter   : 3 特徴量の重要度を知る
# Theme     : 疑似相関における変数重要度
# Created on: 2021/09/20
# Page      : P78 - P82
# URL       : https://github.com/ghmagazine/ml_interpret_book
# ******************************************************************************


# ＜概要＞
# - 変数重要度が高いことを因果関係として解釈してはならない（疑似相関の問題を含むため）
# - PFIはベースラインに対する相対評価のため特に注意が必要


# ＜疑似相関の問題設定＞
# - 実際に影響のある特徴量(X0)と、影響のない特徴量(X1)が存在する
# - X0とX1には相関関係がある
# - モデルにX0が入る場合はX0の重要度が高まる（正しい振舞い）
# - モデルにX0が入らない場合はX1の重要度が高まる（ミスリーディングな振舞い）
#   --- X1が変化してもモデルの予測値は変化するが、売上は増えない


# ＜目次＞
# 0 準備
# 1 シミュレーションデータの生成
# 2 全ての特徴量を用いたモデルの重要度
# 3 最も重要度の高い特徴量を場外したモデルの重要度


# 0 準備 -----------------------------------------------------------------------

# ライブラリ
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

# 自作モジュール
sys.path.append("book/ml_interpret_book")
from module.chap3.data import generate_simulation_data
from module.chap3.data import plot_scatters
from module.chap3.importance import PermutationFeatureImportance


# その他の設定
# --- pandasの有効桁数設定（小数2桁表示）
# --- Seabornの設定
# --- warningsを非表示に
pd.options.display.float_format = "{:.2f}".format
warnings.simplefilter("ignore")


# 1 シミュレーションデータの生成 ----------------------------------------------

# ＜ポイント＞
# - X0-X2の3つの特徴量を作成する
# - X0とX1の相関係数は0.95と高い水準に設定する
# - X0のベータは1となる一方、X1のベータはゼロとなるようにする
#   --- X0とX1は疑似相関の状態となっている


# パラメータ設定
N = 1000
J = 3
mu = np.zeros(J)
Sigma = np.array([[1, 0.95, 0], [0.95, 1, 0], [0, 0, 1]])
beta = np.array([1, 0, 0])

# シミュレーションデータの生成
X_train, X_test, y_train, y_test = \
    generate_simulation_data(N=N, beta=beta, mu=mu, Sigma=Sigma)

# 可視化
var_names = [f"X{j}" for j in range(J)]
plot_scatters(X=X_train, y=y_train, var_names=var_names)


# 2 全ての特徴量を用いたモデルの重要度 ---------------------------------------

# ＜ポイント＞
# - X0の変数重要度のみが高くなり、X1の変数重要度は低くなる
# - 予測精度は高くなっている（ベースラインのRMSEは0.11）

# モデル構築
# --- インスタンス生成
# --- 学習
rf = RandomForestRegressor(n_jobs=-1, random_state=42)
rf.fit(X=X_train, y=y_train)

# 変数重要度の計算
# --- インスタンス生成
# --- 変数重要度の算出
# --- プロット作成
pfi = PermutationFeatureImportance(estimator=rf, X=X_test, y=y_test,
                                   var_names=["X0", "X1", "X2"])
pfi.permutation_feature_importance()
pfi.plot(importance_type="difference")


# 3 最も重要度の高い特徴量を場外したモデルの重要度 -----------------------------

# ＜ポイント＞
# - X0をモデルから除外したケースを考える
# - X1は予測には関係ないはずなのに、X1の変数重要度が高くなってしまう（疑似相関の影響）
# - 予測精度は低下している（ベースラインのRMSEは0.31）

# モデル構築
# --- インスタンス生成
# --- 学習
rf = RandomForestRegressor(n_jobs=-1, random_state=42)
rf.fit(X=X_train[:, [1, 2]], y=y_train)

# 変数重要度の計算
# --- インスタンス生成
# --- 変数重要度の算出
# --- プロット作成
pfi = PermutationFeatureImportance(estimator=rf, X=X_test[:, [1, 2]], y=y_test,
                                   var_names=["X1", "X2"])
pfi.permutation_feature_importance()
pfi.plot(importance_type="difference")


