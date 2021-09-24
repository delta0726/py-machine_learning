# ******************************************************************************
# Book Name : 機械学習を解釈する技術
# Chapter   : 特徴量と予測の関係を知る
# Theme     : 4-5 実データでの分析
# Created on: 2021/09/25
# Page      : P120 - P127
# URL       : https://github.com/ghmagazine/ml_interpret_book
# ******************************************************************************


# ＜概要＞
# - Bostonデータセットにおける住宅価格に対する特徴量の影響を考える
# - PDPは他の特徴量の影響を排除した感応度により関係性を解釈することができる
#   --- 因果関係として解釈してはならない（モデルが正しく関係性を学習している前提が必要）


# ＜目次＞
# 0 準備
# 1 データ準備
# 2 ランダムフォレストによる学習
# 3 PDによる可視化
# 4 散布図による可視化


# 0 準備 -----------------------------------------------------------------------

# ライブラリ
import sys
from functools import partial
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from sklearn.model_selection import train_test_split

# 自作モジュール
sys.path.append("book/ml_interpret_book")
from module.chap4.func import plot_boston_pd


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

# 確認
pprint(vars(rf))


# 3 PDによる可視化 -------------------------------------------------------------

# PDの計算
# --- kind引数はPDの場合は"average"、ICEの場合は"individual"
pdp = partial_dependence(estimator=rf,
                         X=X_test,
                         features=["RM"],
                         kind="average")

# 確認
pdp

# プロット作成
# --- ラグが表示されてデータの密度が確認できる
# --- partial_dependence()で作成したオブジェクトは使用していない
# --- RM： 平均的な部屋の数  DIS：都心からの距離
plot_boston_pd(estimator=rf, X=X_test, var_name=["RM"])
plot_boston_pd(estimator=rf, X=X_test, var_name=["DIS"])


# 4 散布図による可視化 ---------------------------------------------------------

# ＜ポイント＞
# - 散布図を見ることによってもPDPのようなインプリケーションが得れそうな気がする
#   --- 他の特徴量の影響が含まれるのでPDPの方が好ましい

# ＜プロットの解釈＞
# - 都心からの距離が大きくなるほど住宅価格が上昇している
#   --- これは都心に近いほど犯罪率が高く、犯罪率と住宅価格が反比例することによるもの
#   ---


# 関数定義
def plot_lowess():

    lowess_plot = partial(
        sns.regplot,
        lowess=True,
        ci=None,
        scatter_kws={"alpha": 0.3}
    )

    # 3つの散布図を並べて可視化
    fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

    # DIS vs MEDV（都心からの距離と住宅価格）
    lowess_plot(x=X_test["DIS"], y=y_test, ax=axes[0])
    axes[0].set(xlabel="DIS", ylabel="MEDV")

    # DIS vs CRIM（都心からの距離と犯罪率）
    lowess_plot(x=X_test["DIS"], y=np.log(X_test["CRIM"]), ax=axes[1])
    axes[1].set(xlabel="DIS", ylabel="log(CRIM)")

    # CRIM vs MEDV（犯罪率と都心からの距離）
    lowess_plot(x=np.log(X_test["CRIM"]), y=y_test, ax=axes[2])
    axes[2].set(xlabel="log(CRIM)", ylabel="MEDV")

    fig.suptitle("Scatter Plot and Lowess Line")

    fig.show()


# プロット作成
plot_lowess()
