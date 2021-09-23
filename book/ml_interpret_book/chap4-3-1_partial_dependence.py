# ******************************************************************************
# Book Name : 機械学習を解釈する技術
# Chapter   : 特徴量と予測の関係を知る
# Theme     : 4-3 Partial Dependence（イメージ確認）
# Created on: 2021/09/23
# Page      : P98 - P107
# URL       : https://github.com/ghmagazine/ml_interpret_book
# ******************************************************************************


# ＜概要＞
# - インスタンスの特定値のみを変化させた場合の予測値の変化を追跡する
# - Partial Dependence Plotによる予測値の感応度分析の考え方を確認する


# ＜目次＞
# 0 準備
# 1 1つのインスタンスと予測値の関係
# 2 インスタンスと予測値の関係
# 3 特徴量の1つのみを変化させた場合の予測値
# 4 複数の範囲で変化させた場合の予測値
# 5 インスタンスを変更して予測値の変化を確認
# 6 全てのインスタンスに対する特徴量と予測値の平均的な関係


# 0 準備 -----------------------------------------------------------------------

# ライブラリ
import sys
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# 自作モジュール
sys.path.append("book/ml_interpret_book")
from module.chap4.data import generate_simulation_data2

# シミュレーションデータの生成
X_train, X_test, y_train, y_test = generate_simulation_data2()


# 1 1つのインスタンスと予測値の関係 -------------------------------------------------

# モデル構築
rf = RandomForestRegressor()
rf.fit(X=X_train, y=y_train)

# 確認
pprint(vars(rf))


# 2 インスタンスと予測値の関係 ------------------------------------------------------

# パラメータ設定
i = 0

# インスタンスの抽出
Xi = X_test[[i]]
Xi

# 予測値の出力
rf.predict(Xi)


# 3 特徴量の1つのみを変化させた場合の予測値 ------------------------------------------

# 関数定義
# --- 特定のインスタンスを変更して予測値を取得
def counterfactual_prediction(estimator, X, idx_to_replace, value_to_replace):
    """
    :param estimator: 学習済モデル
    :param X: 特徴量
    :param idx_to_replace:値を書き換える特徴量のインデックス
    :type value_to_replace: 置換する値
    :return: 予測値
    """

    # コピー
    # --- 特徴量の上書き防止
    X_replaced = X.copy()

    # 特徴量を置き換えて予測
    X_replaced[:, idx_to_replace] = value_to_replace
    y_pred = estimator.predict(X_replaced)

    return y_pred


# 予測値の取得
# --- -4に置換
cp = counterfactual_prediction(estimator=rf, X=Xi, idx_to_replace=0, value_to_replace=-4)[0]
print(f"(X0, X1)=(-4, 2.15)の時の予測値：{cp: .2f}")

# 予測値の取得
# --- -3に置換
cp = counterfactual_prediction(estimator=rf, X=Xi, idx_to_replace=0, value_to_replace=-3)[0]
print(f"(X0, X1)=(-3, 2.15)の時の予測値：{cp: .2f}")


# 4 複数の範囲で変化させた場合の予測値 ------------------------------------------

# ＜ポイント＞
# - インスタンス0において特徴量X0とモデル予測値の関係を可視化する


# 変化点の作成
X0_range = np.linspace(-np.pi * 2, np.pi * 2, num=50)

# 予測値の作成
cps = np.concatenate(
    [counterfactual_prediction(rf, Xi, 0, x) for x in X0_range]
)


# プロット定義
def plot_line(x, y, xlabel="X", ylabel="Y", title=None):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    fig.suptitle(title)
    plt.show()


# プロット作成
plot_line(x=X0_range, y=cps, xlabel="X0", ylabel="Prediction", title="Model Prediction")


# 5 インスタンスを変更して予測値の変化を確認 ------------------------------------

# ＜ポイント＞
# - 今回はインスタンスを変更しても同じような傾向が見られる
#   --- 各インスタンスでどのような出力になるかは個別に検証する必要がある


# インスタンス10の予測値
i = 10
Xi = X_test[[i]]
rf.predict(Xi)

# 予測値の作成
cps = np.concatenate(
    [counterfactual_prediction(rf, Xi, 0, x) for x in X0_range]
)

# プロット作成
plot_line(x=X0_range, y=cps, xlabel="X0", ylabel="Prediction", title="Model Prediction")


# 6 全てのインスタンスに対する特徴量と予測値の平均的な関係 ------------------------

# ＜ポイント＞
# - 特徴量と予測値の平均的な関係を確認する（インスタンスごとの細かい差異は無視）


# 予測値の平均値
# --- インスタンスごとに予測値のリストを作成してポイントごとの平均値を算出
avg_cps = np.array(
    [counterfactual_prediction(rf, X_test, 0, x).mean() for x in X0_range]
)

# プロット作成
plot_line(x=X0_range, y=avg_cps, xlabel="X0", ylabel="Prediction",
          title="Model Prediction")
