# ******************************************************************************
# Book Name : 機械学習を解釈する技術
# Theme     : 予測の理由を考える
# Chapter   : SHAPの実装（関数によるイメージ）
# Created on: 2021/09/29
# Page      : P187 - P195
# URL       : https://github.com/ghmagazine/ml_interpret_book
# ******************************************************************************


# ＜概要＞
# - SHAPをクラスに実装する


# ＜目次＞
# 0 準備
# 1 SHAPのクラス実装
# 2 SHAPの計算


# 0 準備 ----------------------------------------------------------------------

import sys
from dataclasses import dataclass
from itertools import combinations
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import factorial
from sklearn.ensemble import RandomForestRegressor

# 自作モジュール
from module.chap6.data import generate_simulation_data

# その他の設定
# --- ランダムシードの設定
np.random.seed(42)

# シミュレーションデータの生成
X_train, X_test, y_train, y_test = generate_simulation_data()

# モデル構築
rf = RandomForestRegressor(n_jobs=-1, random_state=24)
rf.fit(X=X_train, y=y_train)


# 1 SHAPのクラス実装 ------------------------------------------------------------

@dataclass
class ShapleyAdditiveExplanations:
    estimator: Any
    X: np.ndarray
    var_names: List[str]

    def __post_init__(self) -> None:
        # ベースラインとしての平均的な予測値
        self.baseline = self.estimator.predict(self.X).mean()

        # 特徴量の総数
        self.J = self.X.shape[1]

        # 特徴量の組み合わせパターン
        self.subsets = [
            s
            for j in range(self.J + 1)
            for s in combinations(range(self.J), j)
        ]

    def _get_expected_value(self, subset: Tuple[int, ...]) -> np.ndarray:
        # 元データをコピーして使用
        # --- 上書き防止
        _X = self.X.copy()

        #
        if subset is not None:
            _s = list(subset)
            _X[:, _s] = _X[self.i, _s]

        return self.estimator.predict(_X).mean()

    def _calc_weighted_marginal_contribution(self, J: int, s_union_j: Tuple[int, ...]):

        # 特徴量Jがない場合の組み合わせ
        s = tuple(set(s_union_j) - set([J]))

        # 組み合わせの数
        S = len(s)

        # 組み合わせの出現回数
        weight = factorial(S) * factorial(self.J - S - 1)

        # 限界貢献度
        marginal_contribution = (
            self.expected_values[s_union_j] - self.expected_values[s]
        )

        return weight * marginal_contribution

    def shapley_additive_explanations(self, id_to_compute: int) -> None:
        # インスタンス番号の格納
        self.i = id_to_compute

        # 予測値の計算
        # --- 全ての組み合わせに対して行う
        self.expected_values = {
            s: self._get_expected_value(s) for s in self.subsets
        }

        # SHAP値の計算
        shap_values = np.zeros(self.J)
        for j in range(self.J):
            shap_values[j] = np.sum([
                self._calc_weighted_marginal_contribution(j, s_union_j)
                for s_union_j in self.subsets
                if j in s_union_j
            ]) / factorial(self.J)

        # データフレームとしてまとめる
        self.df_shap = pd.DataFrame(
            data={
                "var_name": self.var_names,
                "feature_value": self.X[id_to_compute],
                "shap_value": shap_values
            }
        )

    def plot(self):
        # データフレームのコピー
        # --- 上書き防止
        df = self.df_shap.copy()

        # ラベル作成
        df['label'] = [
            f"{x} = {y: .2f}" for x, y in zip(df.var_name, df.feature_value)
        ]

        # SHAP値が高い順に位並び替え
        df = df.sort_values("shap_value").reset_index(drop=True)

        # 全体の特徴量が分かっている時の予測値
        predicted_value = self.expected_values[self.subsets[-1]]

        # 棒グラフによる可視化
        fig, ax = plt.subplots()
        ax.barh(df.label, df.shap_value)
        ax.set(xlabel="SHAP", ylabel=None)
        fig.suptitle(f"SHAP \n(Baseline: {self.baseline:.2f}, Prediction: {predicted_value:.2f}, Difference: {predicted_value - self.baseline:.2f})")

        # プロット出力
        fig.show()


# 2 SHAPの計算 ---------------------------------------------------------------

# インスタンス生成
shap = ShapleyAdditiveExplanations(estimator=rf, X=X_test, var_names=["X0", "X1"])

# SHAP値の計算
shap.shapley_additive_explanations(id_to_compute=1)

# SHAP値の出力
shap.df_shap

# SHAP値の可視化
shap.plot()

# デバッグ用
# self = shap
