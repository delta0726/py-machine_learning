# ******************************************************************************
# Book Name : 機械学習を解釈する技術
# Chapter   : インスタンスごとの特異性をとらえる
# Theme     : 5-6 実データでの分析
# Created on: 2021/09/27
# Page      : P159 - P163
# URL       : https://github.com/ghmagazine/ml_interpret_book
# ******************************************************************************


# ＜概要＞


# ＜目次＞
# 0 準備
# 1 データ準備
# 2 ランダムフォレストによる学習
# 3 PDとICEによる解釈


# 0 準備 -----------------------------------------------------------------------

# ライブラリ

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence
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


# 3 PDとICEによる解釈 -------------------------------------------------------------

# PDとICEの計算
ice = partial_dependence(estimator=rf, X=X_test, features=["RM"], kind="both")
ice


# プロット定義
def plot_ice():
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_partial_dependence(estimator=rf, X=X_test, features=["RM"],
                            kind="both", ax=ax)
    fig.show()


# プロット作成
plot_ice()

