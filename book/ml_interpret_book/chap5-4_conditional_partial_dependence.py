# ******************************************************************************
# Book Name : 機械学習を解釈する技術
# Chapter   : インスタンスごとの特異性をとらえる
# Theme     : 5-4 Conditional Partial Dependence
# Created on: 2021/09/26
# Page      : P148 - P152
# URL       : https://github.com/ghmagazine/ml_interpret_book
# ******************************************************************************


# ＜概要＞
# - CPDによる交互作用への対応策を知る


# ＜目次＞
# 0 準備
# 1 CPDの可視化


# 0 準備 ----------------------------------------------------------------------

import sys

from sklearn.ensemble import RandomForestRegressor

from module.chap5.pdp import PartialDependence
from module.chap5.ice import IndividualConditionalException
from module.chap5.data import generate_simulation_data


# シミュレーションデータの生成
X_train, X_test, y_train, y_test = generate_simulation_data()

# モデル構築
rf = RandomForestRegressor(n_jobs=-1, random_state=42)
rf.fit(X=X_train, y=y_train)


# 1 CPDの可視化 ---------------------------------------------------------------

# ＜ポイント＞
# - CPDとはインスタンスが特定の条件におけるPDをいう
#   --- CPD: Conditional Partial Dependence
#   --- 交互作用をうまく分離できれば、PDで平均化されることによる欠点を補うことができる
#   --- 実際にはグルーピングの条件指定を事前に知ることはできないので機能しないことが多い


# X2が0の場合（右下がり） ---------------------------------

# インスタンス生成
pdp = PartialDependence(estimator=rf, X=X_test[X_test[:, 2] == 0],
                        var_names=["X0", "X1", "X2"])

# PDの計算
pdp.partial_dependence("X1")

# プロット作成
pdp.plot()


# X2が1の場合（右上がり） ---------------------------------

# インスタンス生成
pdp = PartialDependence(estimator=rf, X=X_test[X_test[:, 2] == 1],
                        var_names=["X0", "X1", "X2"])

# PDの計算
pdp.partial_dependence("X1")

# プロット作成
pdp.plot()


# ids_to_compute:1-20の場合 --------------------------------------

# インスタンス生成
ice = IndividualConditionalException(estimator=rf, X=X_test,
                                     var_names=["X0", "X1", "X2"])

# ICEの計算
# --- 特徴量：X1 / インスタンス： 1
ice.individual_conditional_exception(var_name="X1", ids_to_compute=range(20))

# 予測値の出力
ice.df_instance

# ICEのプロット
ice.plot(ylim=(-6, 6))