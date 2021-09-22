# ******************************************************************************
# Book Name : 機械学習を解釈する技術
# Chapter   : 3 特徴量の重要度を知る
# Module    : simulation_data.py
# Created on: 2021/09/18
# Page      : P61 - P69
# URL       : https://github.com/ghmagazine/ml_interpret_book
# ******************************************************************************


import warnings
from dataclasses import dataclass, field
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

warnings.simplefilter("ignore")


@dataclass
class PermutationFeatureImportance:

    # 変数定義
    estimator: Any
    X: np.ndarray
    y: np.ndarray
    var_names: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        ベースラインのRMSEを算出（シャッフルなしのRMSE）
        return: None
        """
        self.baseline = mean_squared_error(
            y_true=self.y,
            y_pred=self.estimator.predict(self.X),
            squared=False
        )

    def _permutation_metrics(self, idx_to_permute: int) -> float:
        """
       特徴量をシャッフルした場合の予測精度
       :param idx_to_permute:
       :type idx_to_permute: int
       :return:
       :rtype: float
       """

        # コピー
        # --- 元のオブジェクトの上書きを防止
        X_permuted = self.X.copy()

        # 特定の特徴量をシャッフル
        # --- ランダム化したと特定列のデータを再入力
        X_permuted[:, idx_to_permute] = np.random.permutation(
            X_permuted[:, idx_to_permute]
        )

        # 予測
        y_pred = self.estimator.predict(X_permuted)

        # 出力
        return mean_squared_error(y_true=self.y, y_pred=y_pred, squared=False)

    # PFIの算出
    def permutation_feature_importance(self, n_shuffle: int = 10) -> None:
        # 特徴量の数
        J = self.X.shape[1]

        metrics_permuted = [
            np.mean(
                [self._permutation_metrics(j) for _ in range(n_shuffle)]
            )
            for j in range(J)
        ]

        # データフレームとしてまとめる
        df_feature_importance = pd.DataFrame(
            data={
                "var_name": self.var_names,
                "baseline": self.baseline,
                "permutation": metrics_permuted,
                "difference": metrics_permuted - self.baseline,
                "ratio": metrics_permuted / self.baseline
            }
        )

        self.feature_importance = df_feature_importance.sort_values(
            "permutation", ascending=False
        )

    def plot(self, importance_type: str = "difference") -> None:
        fig, ax = plt.subplots()
        ax.barh(
            self.feature_importance["var_name"],
            self.feature_importance[importance_type],
            label=f"baseline: {self.baseline:.2f}"
        )
        ax.set(xlabel=importance_type)
        ax.invert_yaxis()
        ax.legend(loc="lower right")
        fig.suptitle(f"Permutationによる特徴量の重要度({importance_type})")
        fig.show()
