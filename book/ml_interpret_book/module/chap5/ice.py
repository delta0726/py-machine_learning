from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 自作モジュール
from book.ml_interpret_book.module.chap5.pdp import PartialDependence


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
