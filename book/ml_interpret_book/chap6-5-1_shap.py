# ******************************************************************************
# Book Name : 機械学習を解釈する技術
# Theme     : 5-6 実データでの分析
# Chapter   : 予測の理由を考える
# Created on: 2021/09/27
# Page      : P182 - P187
# URL       : https://github.com/ghmagazine/ml_interpret_book
# ******************************************************************************


# ＜概要＞
# - 限界貢献度とSHAPの考え方を整理する


# ＜目次＞
# 0 準備
# 1 シミュレーションデータの生成
# 2 予測データフレームの作成
# 3 SHAPの発想に基づく予測値
# 4 SHAPの算出


# 0 準備 ----------------------------------------------------------------------

import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 自作モジュール
sys.path.append("book/ml_interpret_book")
from mli.metrics import regression_metrics

# その他の設定
# --- ランダムシードの設定
np.random.seed(42)


# 1 シミュレーションデータの生成 -------------------------------------------------

# ＜ポイント＞
# - SHAPはコンセプトが複雑なので単純なデータで実装を確認する
#   --- 特徴量は２つのみ（X0：影響なし / X1：影響ありの）


# シミュレーションデータの定義
def generate_simulation_data():

    # パラメータ設定
    # --- インスタンス数
    # --- 列数
    N = 1000
    J = 2

    # 傾き
    # --- X0：影響なし X1：影響あり
    beta = np.array([0, 1])

    # 特徴量の生成
    X = np.random.normal(loc=0, scale=1, size=[N, J])
    e = np.random.normal(loc=0, scale=0.1, size=N)

    # 線形和で目的変数を作成
    y = X @ beta + e

    return train_test_split(X, y, test_size=0.2, random_state=42)


# シミュレーションデータの生成
X_train, X_test, y_train, y_test = generate_simulation_data()


# 2 予測データフレームの作成 ---------------------------------------------

# モデル構築
rf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# 予測精度の確認
regression_metrics(estimator=rf, X=X_test, y=y_test)

# データフレーム作成
# --- 目的変数
# --- 予測値の追加
# --- ベースラインの追加
df = pd.DataFrame(data=X_test, columns=["X0", "X1"])
df["y_pred"] = rf.predict(X_test)
df["y_pred_baseline"] = rf.predict(X_test).mean()

# データフレーム確認
df.head()


# 3 SHAPの発想に基づく予測値の算出------------------------------------------

# インスタンス1の抽出
x = X_test[1, :]

# * Case1：X0とX1がともに分かっていない場合の予測値 ----------------

# 予測の平均値
E_baseline = rf.predict(X_test).mean()


# * Case2：X0のみが分かっていない場合の予測値 ---------------------

# 全てのX0をインスタンス1のX0に置換
X0 = X_test.copy()
X0[:, 0] = x[0]

# 予測値の平均
E0 = rf.predict(X0).mean()


# * Case3：X1のみが分かっていない場合の予測値 ---------------------

# 全てのX1をインスタンス1のX1に置換
X1 = X_test.copy()
X1[:, 1] = x[1]

# 予測値の平均
E1 = rf.predict(X1).mean()


# * Case4：X1とX2がそれぞれ分かっている場合の予測値 ---------------------

# インスタンス1のみの予測値
E_full = rf.predict(x[np.newaxis, :])[0]


# * まとめ --------------------------------------------------------

# 結果を出力
print(f"CASE1: X0もX1も分かっていないときの予測値 -> {E_baseline: .2f}")
print(f"CASE2: X0のみが分かっているときの予測値 -> {E0: .2f}")
print(f"CASE3: X1のみが分かっているときの予測値 -> {E1: .2f}")
print(f"CASE4: X1もX2も分かっているときの予測値 -> {E_full: .2f}")


# 4 SHAPの算出 --------------------------------------------------------------

# ＜ポイント＞
# - 予測値(0.68)とベースラインの差(0.05)である0.63をSHAPで寄与度分解する
#   --- 0.68 - 0.05 = 0.02 + 0.62


# SHAPの算出
SHAP0 = ((E0 - E_baseline) + (E_full - E1)) / 2
SHAP1 = ((E1 - E_baseline) + (E_full - E0)) / 2

# 確認
print(f"(SHAP0, SHAP1) = {SHAP0:.2f}, {SHAP1:.2f}")
