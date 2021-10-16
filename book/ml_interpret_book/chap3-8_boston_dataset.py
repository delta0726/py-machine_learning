# ******************************************************************************
# Book Name : 機械学習を解釈する技術
# Chapter   : 3 特徴量の重要度を知る
# Theme     : 実データでの分析
# Created on: 2021/09/20
# Page      : P83 - P86
# URL       : https://github.com/ghmagazine/ml_interpret_book
# ******************************************************************************


# ＜概要＞
# - Bostonデータセットにおけるランダムフォレストモデルの変数重要度を算出する
# - {sklearn.inspection}で提供されるモジュールを使用


# ライブラリ
import sys
from pprint import pprint

import pandas as pd
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# 自作モジュール
from module.chap3.func import plot_bar


# 1 データ準備 -----------------------------------------------------------------

# データセットのロード
boston = load_boston()

# データ格納
X = pd.DataFrame(data=boston["data"], columns=boston["feature_names"])
y = boston["target"]

# データ分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 2 ランダムフォレストによる学習 -------------------------------------------------

# モデル構築
# --- インスタンスの生成
# --- 学習
rf = RandomForestRegressor(n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# 確認
pprint(vars(rf))


# 3 変数重要度の算出 ----------------------------------------------------------

# PFIの算出
pdi = permutation_importance(estimator=rf,
                             X=X_test,
                             y=y_test,
                             scoring="neg_root_mean_squared_error",
                             n_repeats=5,
                             n_jobs=1,
                             random_state=42)

# オブジェクト確認
# --- 辞書型で格納される
pdi.keys()
pprint(pdi)

# データフレーム格納
df_pfi = pd.DataFrame(
    data={"var_name": X_test.columns,
          "importance": pdi["importances_mean"]}
).sort_values("importance")

# データ確認
df_pfi


# 4 変数重要度の可視化 --------------------------------------------------------

# プロット作成
plot_bar(df_pfi["var_name"],
         df_pfi["importance"],
         xlabel="difference",
         title="Permutation Importance (diffferemnce)")
