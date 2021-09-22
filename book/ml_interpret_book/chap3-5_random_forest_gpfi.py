# ******************************************************************************
# Book Name : 機械学習を解釈する技術
# Chapter   : 3 特徴量の重要度を知る
# Theme     : PFIによるランダムフォレストの変数重要度の算出
# Created on: 2021/09/18
# Page      : P70 - P78
# URL       : https://github.com/ghmagazine/ml_interpret_book
# ******************************************************************************


# ＜概要＞
# - 相関が高い特徴量はグループ化して変数重要度を測定した方がよい
# - カテゴリカル変数をOne-Hotエンコーディングする場合はグループ化したほうが解釈性が高まる
#   --- ツリー系モデルの場合はカテゴリカル変数のまま扱うほうがよい


# ＜クラスの継承＞
# - PermutationFeatureImportanceののinit()とplot()を継承により活用
# - permutation_feature_importance()のメイン部分は独自のメソッドを作成


# ＜目次＞
# 0 準備
# 1 シミュレーションデータの生成
# 2 Grouped Feature Importanceの実装
# 3 ランダムフォレストモデルの作成


# 0 準備 -----------------------------------------------------------------------

# ライブラリ
import sys
import warnings

import numpy as np
import pandas as pd
from typing import List

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# 自作モジュール
sys.path.append("book/ml_interpret_book")
from module.chap3.data import generate_simulation_data
from module.chap3.data import plot_scatters
from module.chap3.importance import PermutationFeatureImportance


# その他の設定
# --- pandasの有効桁数設定（小数2桁表示）
# --- Seabornの設定
# --- warningsを非表示に
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


# 2 Grouped Feature Importanceの実装 --------------------------------------

class GroupedFeatureImportance(PermutationFeatureImportance):

    def _permutation_metrics(self,
                             var_names_to_permute: List[str]
                             ) -> float:

        # データコピー
        X_permuted = self.X.copy()

        # 特徴量名をインデックスに変換
        idx_to_permute = [self.var_names.index(v) for v in var_names_to_permute]

        # 特徴量群をまとめてシャッフル
        X_permuted[:, idx_to_permute] = np.random.permutation(
            X_permuted[:, idx_to_permute]
        )

        # 予測値の作成
        y_pred = self.estimator.predict(X_permuted)

        # 予測精度の計算
        return mean_squared_error(y_true=self.y, y_pred=y_pred)

    def permutation_feature_importance(self,
                                       var_groups: List[List[str]] = None,
                                       n_shuffle: int = 10
                                       ) -> None:

        # グループが指定されない場合は1グループ
        if var_groups is None:
            var_groups = [[j] for j in self.var_names]

        # グループごとの重要度を計算
        metric_permuted = [
            np.mean([self._permutation_metrics(j) for _ in range(n_shuffle)])
            for j in var_groups
        ]

        # データフレーム作成
        df_feature_importance = pd.DataFrame(
            data={
                "var_name": ["+".join(j) for j in var_groups],
                "baseline": self.baseline,
                "permutation": metric_permuted,
                "difference": metric_permuted - self.baseline,
                "ratio": metric_permuted / self.baseline
            }
        )

        # 並び替え
        self.feature_importance = df_feature_importance.sort_values(
            "permutation", ascending=False
        )


# 3 ランダムフォレストモデルの作成 -----------------------------------------

# データ作成
# --- X2と全く同じ特徴量を追加
X_train2 = np.concatenate([X_train, X_train[:, [2]]], axis=1)
X_test2 = np.concatenate([X_test, X_test[:, [2]]], axis=1)

# データ確認
X_train2[0:4, :]

# 学習
# --- ランダムフォレスト
rf = RandomForestRegressor(n_jobs=-1, random_state=42)
rf.fit(X=X_train2, y=y_train)

# オブジェクト確認
vars(rf)

# 予測精度の確認
# --- R2=0.993
r2_score(y_true=y_test, y_pred=rf.predict(X_test2))

# 変数重要度の計算
# --- インスタンス生成
# --- 変数重要度の計算
gpfi = GroupedFeatureImportance(estimator=rf, X=X_test2, y=y_test,
                                var_names=["X0", "X1", "X2", "X3"])
gpfi.permutation_feature_importance()
gpfi.plot()

# デバッグ
# self = gpfi
