# ******************************************************************************
# Book Name : 機械学習を解釈する技術
# Chapter   : 予測の理由を考える
# Theme     : 6-7 ミクロからマクロへ
# Created on: 2021/09/29
# Page      : P201 - P207
# URL       : https://github.com/ghmagazine/ml_interpret_book
# ******************************************************************************


# ＜概要＞
# - shapは適当な粒度に集計することでマクロ視点での解釈手法として用いることもできる
#   --- 協力ゲーム理論を背景とした寄与度分解に望ましい性質を持っている
#   --- 協力ゲーム理論は直感的でない部分があるので非専門家への説明が難しい場合もある


# ＜目次＞
# 0 準備
# 1 データ準備
# 2 ランダムフォレストによる学習
# 3 SHAPインスタンスの作成
# 4 SHAPによる変数重要度
# 5 SHAPによるPDP


# 0 準備 -----------------------------------------------------------------------

import sys

import pandas as pd
import shap
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 自作モジュール
from module.chap6.data import generate_simulation_data_2


# 1 データ準備 -----------------------------------------------------------------

# データセットのロード
boston = load_boston()

# データ格納
X = pd.DataFrame(data=boston["data"], columns=boston["feature_names"])
y = boston["target"]

# データ分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 2 ランダムフォレストによる学習 -------------------------------------------------

# モデル構築
# --- インスタンスの生成
# --- 学習
rf = RandomForestRegressor(n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)


# 3 SHAPインスタンスの作成 ------------------------------------------------------

# Explainerの定義
# --- インスタンスの作成
explailer = shap.TreeExplainer(model=rf, data=X_test,
                               feature_perturbation="interventional")

# SHAPの計算
shap_values = explailer(X_test)

# 確認
vars(explailer)
vars(shap_values)


# 4 SHAPによる変数重要度 -------------------------------------------------------

# ＜ポイント＞
# - インスタンスごとのshpa値を絶対値で合計することで算出
# - SHAP値を個別に集計するため分布を用いた分析が可能となる
#   --- ミクロから積み上げることでトップダウンのPFIより柔軟な分析が可能

# 変数重要度（マクロレベル）
shap.plots.bar(shap_values=shap_values)

# SHAP値の分布
shap.plots.beeswarm(shap_values=shap_values)


# 5 SHAPによるPDP ------------------------------------------------------------

# シミュレーションデータの生成
X_train, X_test, y_train, y_test = generate_simulation_data_2()

# モデル構築
rf2 = RandomForestRegressor(n_jobs=-1, random_state=42)
rf2.fit(X=X_train, y=y_train)

# 予測値のデータフレーム作成
X_test = pd.DataFrame(data=X_test, columns=["X0", "X1", "X2"])

# Explainerの作成
explailer2 = shap.TreeExplainer(model=rf2, data=X_test)

# SHAP値の計算
shap_values_2 = explailer2(X_test)

# PDPの作成
shap.plots.scatter(shap_values_2[:, "X1"], color=shap_values_2)
