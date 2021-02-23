# ******************************************************************************
# Chapter   : 2 モデル構築前のワークフローと前処理
# Title     : 2-6 さまざまな戦略を使って欠損値を補完する（Recipe14)
# Created by: Owner
# Created on: 2020/12/24
# Page      : P57 - P61
# ******************************************************************************

# ＜概要＞
# - 欠損値の補完は単純な補完方法に加えて、アルゴリズムを用いる補完方法もある
#   --- ここでは単純な処理のみを扱う
# - データセットと適合する再利用可能なクラスを生成していく（scikit-learn全体のテーマでもある）


# ＜目次＞
# 0 準備
# 1 単純な欠損値補完
# 2 変換パターン
# 3 特定の値に置換


# 0 準備 -----------------------------------------------------------------------------------

import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.impute import SimpleImputer


# データ準備
iris = datasets.load_iris()

# データ構造
iris.keys()

# データ格納
iris_X = iris.data

# 欠損値を挿入
# --- iris_X
# --- 二項乱数で生成したTrue(1)の場所にNaNを挿入
masking_array = np.random.binomial(n=1, p=0.25, size=iris_X.shape).astype(bool)
iris_X[masking_array] = np.nan
iris_X[:5]


# 1 単純な欠損値補完 --------------------------------------------------------------------------

# インスタンス生成
impute = SimpleImputer(strategy='mean')

# 適用
# --- 列ごとに補完
iris_X_prime = impute.fit_transform(iris_X)
iris_X_prime[:5]

# 元データと比較
iris_X[:5]


# 2 変換パターン --------------------------------------------------------------------------------

# ＜ポイント＞
# - 変換方法(strategy)には以下の方法がある
#   --- "mean", "median", "most_frequent", "constant"

# インスタンスの再生成
# --- medianを選択
impute = SimpleImputer(strategy='median')

# 適用
iris_X_prime = impute.fit_transform(iris_X)
iris_X_prime[:5]


# 3 特定の値に置換 --------------------------------------------------------------------------------

# ＜ポイント＞
# - 再利用の観点ではSimpleImputerクラスが優れている
# - pandasの欠損値補完はテンポラリな使用の柔軟性が高い（再利用性は劣る）

# インデックスを用いて置換
iris_X[np.isnan(iris_X)] = -1
iris_X[:5]

# SimpleImputerクラスを用いて置換
impute = SimpleImputer(missing_values=-1)
impute.fit_transform(iris_X)
iris_X[:5]

# 条件文を用いて置換
# --- pandasの処理で欠損値の判定がしやすい
iris_X_prime = np.where(pd.DataFrame(iris_X).isnull(), -1, iris_X)

# fillna()の利用
# --- valuesプロパティでarrayを抽出
pd.DataFrame(iris_X).fillna(-1)[:5].values
