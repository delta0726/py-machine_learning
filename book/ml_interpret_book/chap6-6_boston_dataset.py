# ******************************************************************************
# Book Name : 機械学習を解釈する技術
# Chapter   : 予測の理由を考える
# Theme     : 6-6 実データでの分析
# Created on: 2021/09/29
# Page      : P196 - P200
# URL       : https://github.com/ghmagazine/ml_interpret_book
# ******************************************************************************


# ＜概要＞
# - 実務ではshapパッケージを用いることで高速に計算することが可能
# - インスタンスごとの寄与度分解をウォーターフォールプロットで確認する


# ＜目次＞
# 0 準備
# 1 データ準備
# 2 ランダムフォレストによる学習
# 3 SHAPの計算
# 4 SHAPの可視化


# 0 準備 -----------------------------------------------------------------------

import pandas as pd
import shap
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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


# 3 SHAPの計算 ----------------------------------------------------------------

# Explainerの定義
# --- インスタンスの作成
explailer = shap.TreeExplainer(model=rf, data=X_test,
                               feature_perturbation="interventional")

# SHAPの計算
shap_values = explailer(X_test)

# インスタンス0の情報
shap_values[0]

# ＜参考＞
# オブジェクトの中身を確認
explailer.__dict__.keys()
shap_values.__dict__.keys()


# 4 SHAPの可視化 -------------------------------------------------------------

# ＜ポイント＞
# - ベースライン(予測値の平均)とインスタンス予測値の差分を寄与度分解している
#   --- インスタンスごとの予想値と実績値の差分ではないことに注意

# 可視化
shap.plots.waterfall(shap_values[0])
shap.plots.waterfall(shap_values[1])
