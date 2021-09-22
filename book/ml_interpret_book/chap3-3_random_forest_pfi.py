# ******************************************************************************
# Book Name : 機械学習を解釈する技術
# Chapter   : 3 特徴量の重要度を知る
# Theme     : PFIによるランダムフォレストの変数重要度の算出
# Created on: 2021/09/18
# Page      : P61 - P69
# URL       : https://github.com/ghmagazine/ml_interpret_book
# ******************************************************************************


# ＜概要＞
# - Permutation Feature Importanceを実装する
# - ランダムフォレストをPFIを用いて変数重要度を算出する
# - dataclassを用いたクラス定義の方法を学ぶ


# ＜dataclass＞
# Python3.7で導入されたdataclass入門
# https://myenigma.hatenablog.com/entry/2020/03/07/171015


# ＜目次＞
# 0 準備
# 1 シミュレーションデータの生成
# 2 PFIの実装
# 3 ランダムフォレストモデルの作成


# 0 準備 -----------------------------------------------------------------------

# ライブラリ
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# 自作モジュール
sys.path.append("book/ml_interpret_book")
from module.chap3.data import generate_simulation_data
from module.chap3.data import plot_scatters


# その他の設定
# --- ランダムシードの設定
# --- pandasの有効桁数設定（小数2桁表示）
# --- Seabornの設定
# --- warningsを非表示に
np.random.seed(42)
pd.options.display.float_format = "{:.2f}".format
warnings.simplefilter("ignore")


# 1 シミュレーションデータの生成 ----------------------------------------------

# パラメータ設定
N = 1000
J = 3
mu = np.zeros(J)
Sigma = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
beta = np.array([0, 1, 2])

# シミュレーションデータの生成
X_train, X_test, y_train, y_test = \
    generate_simulation_data(N=N, beta=beta, mu=mu, Sigma=Sigma)

# 可視化
var_names = [f"X{j}" for j in range(J)]
plot_scatters(X=X_train, y=y_train, var_names=var_names)


# 2 PFIの実装 ---------------------------------------------------------

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


# 3 ランダムフォレストモデルの作成 -----------------------------------------

# 学習
# --- ランダムフォレスト
rf = RandomForestRegressor(n_jobs=-1, random_state=42)
rf.fit(X=X_train, y=y_train)

# オブジェクト確認
vars(rf)

# 予測精度の確認
# --- R2=0.988
r2_score(y_true=y_test, y_pred=rf.predict(X_test))

# 変数重要度の確認
# --- インスタンス生成
# --- PFIの計算
# --- PFIのプロット
pfi = PermutationFeatureImportance(estimator=rf, X=X_test, y=y_test, var_names=var_names)
pfi.permutation_feature_importance()
pfi.plot()

# デバッグ
# self = pfi

