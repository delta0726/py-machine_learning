# ******************************************************************************
# Book Name : 機械学習を解釈する技術
# Chapter   : 特徴量と予測の関係を知る
# Theme     : 4-3 Partial Dependence（クラス実装）
# Created on: 2021/09/23
# Page      : P107 - P110
# URL       : https://github.com/ghmagazine/ml_interpret_book
# ******************************************************************************


from dataclasses import dataclass
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class PartialDependence:
    estimator: Any
    X: np.ndarray
    var_names: List[str]

    def _counterfactual_prediction(self,
                                   idx_to_replace: int,
                                   value_to_replace: float) -> np.ndarray:

        # コピー
        # --- 上書き防止
        X_replaced = self.X.copy()

        # 特徴量の値を置き換えて予測
        X_replaced[:, idx_to_replace] = value_to_replace
        y_pred = self.estimator.predict(X_replaced)

        return y_pred

    def partial_dependence(self,
                           var_name: str,
                           n_grid: int = 50) -> None:

        # 変数名の保存
        # --- 可視化の際に使用
        self.target_var_name = var_name

        # ターゲット特徴量のインデックスを取得
        var_index = self.var_names.index(var_name)

        # 線形グリッドの作成
        # --- ターゲット特徴量の最小値/最大値からNグリッドを作成
        value_range = np.linspace(
            self.X[:, var_index].min(),
            self.X[:, var_index].max(),
            num=n_grid
        )

        # インスタンスごとのモデルの予測値を平均
        average_prediction = np.array([
            self._counterfactual_prediction(var_index, x).mean()
            for x in value_range
        ])

        # 結果の格納
        self.df_partial_dependence = pd.DataFrame(
            data={var_name: value_range,
                  "avg_pred": average_prediction}
        )

        return self.df_partial_dependence

    def plot(self) -> None:
        fig, ax = plt.subplots()
        ax.plot(
            self.df_partial_dependence[self.target_var_name],
            self.df_partial_dependence["avg_pred"]
        )

        ax.set(
            xlabel=self.target_var_name,
            ylabel="Average Prediction",
            ylim=None
        )

        fig.suptitle(f"Partial Dependence Plot ({self.target_var_name})")

        fig.show()
