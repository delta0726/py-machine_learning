# ******************************************************************************
# Book Name : 機械学習を解釈する技術
# Chapter   : インスタンスごとの特異性をとらえる
# Theme     : 5-3 Individual Conditional Exception
# Created on: 2021/09/26
# Page      : P140 - P147
# URL       : https://github.com/ghmagazine/ml_interpret_book
# ******************************************************************************


# ＜概要＞
# - ICEを実装してPDの限界に対するソリューションを探る


# ＜目次＞
# 0 準備
# 1 ICEの実装
# 2 シミュレーションデータへのICEの適用


# 0 準備 --------------------------------------------------------------------

import sys
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# 自作モジュール
from module.chap5.pdp import PartialDependence
from module.chap5.data import generate_simulation_data

# シミュレーションデータの準備
X_train, X_test, y_train, y_test = generate_simulation_data()

# ランダムフォレストモデルの生成
rf = RandomForestRegressor(n_jobs=-1, random_state=42)
rf.fit(X=X_train, y=y_train)


# 1 ICEの実装 ---------------------------------------------------------------

@dataclass
class IndividualConditionalException(PartialDependence):

    def individual_conditional_exception(
            self,
            var_name: str,
            ids_to_compute: List[int],
            n_grid=50
    ) -> None:
        # 変数名の保存
        # --- 可視化の際に使用
        self.target_var_name = var_name

        # 変数名に対応するインデックスの取得
        var_index = self.var_names.index(var_name)

        # 線形グリッドの作成
        # --- ターゲット特徴量の最小値/最大値からNグリッドを作成
        value_range = np.linspace(
            self.X[:, var_index].min(),
            self.X[:, var_index].max(),
            num=n_grid
        )

        # モデル予測値
        # --- インスタンスごと
        individual_prediction = np.array([
            self._counterfactual_prediction(var_index, x)[ids_to_compute]
            for x in value_range
        ])

        # データフレームを作成
        # --- ICEをまとめる
        self.df_ice = \
            pd.DataFrame(data=individual_prediction, columns=ids_to_compute) \
                .assign(**{var_name: value_range}) \
                .melt(id_vars=var_name, var_name="instance", value_name="ice")

        self.df_instance = \
            pd.DataFrame(data=self.X[ids_to_compute], columns=self.var_names) \
                .assign(instance=ids_to_compute,
                        prediction=self.estimator.predict(self.X[ids_to_compute])) \
                .loc[:, ["instance", "prediction"] + self.var_names]

    def plot(self, ylim: List[float]) -> None:
        fig, ax = plt.subplots()

        # ラインプロット
        # --- ICEの線
        sns.lineplot(
            self.target_var_name,
            "ice",
            units="instance",
            data=self.df_ice,
            lw=0.8,
            alpha=0.5,
            estimator=None,
            zorder=1,
            ax=ax
        )

        # プロット作成
        # --- インスタンスから実際の予測点でプロット
        sns.scatterplot(
            self.target_var_name,
            "predicti"
            "on",
            data=self.df_instance,
            zorder=2,
            ax=ax
        )

        ax.set(xlabel=self.target_var_name, ylabel="Prediction", ylim=ylim)
        fig.suptitle(f"Individual Conditional Exception ({self.target_var_name})")

        fig.show()


# 2 シミュレーションデータへのICEの適用 ---------------------------------------

# インスタンス生成
ice = IndividualConditionalException(estimator=rf, X=X_test,
                                     var_names=["X0", "X1", "X2"])

# デバッグ用
# self = ice


# ids_to_compute：0 --------------------------------

# ICEの計算
# --- X1 / インスタンス0
ice.individual_conditional_exception(var_name="X1", ids_to_compute=[0])

# 出力
ice.df_instance

# プロット作成
ice.plot(ylim=(-6, 6))


# ids_to_compute：1 --------------------------------

# ICEの計算
# --- X1 / インスタンス1
ice.individual_conditional_exception(var_name="X1", ids_to_compute=[1])

# 出力
ice.df_instance

# プロット作成
ice.plot(ylim=(-6, 6))


# ids_to_compute：0-20 --------------------------------

# ICEの計算
# --- X1 / インスタンス1
ice.individual_conditional_exception(var_name="X1", ids_to_compute=range(20))

# 出力
ice.df_instance

# プロット作成
ice.plot(ylim=(-6, 6))

