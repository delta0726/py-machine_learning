# ---------------------------------------------------------------------------------------------------
# Library   : Scikit-Learn
# Category  : preprocessing
# Function  : scale
# Created by: Owner
# Created on: 2021/5/11
# URL       : https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html#sklearn.preprocessing.scale
# ---------------------------------------------------------------------------------------------------


# ＜概要＞
# - データセットをZスコア変換する


# ＜構文＞
# sklearn.preprocessing.scale(X, *, axis=0, with_mean=True, with_std=True, copy=True)


# ＜引数＞
# axis: 0(列方向/特徴量) / 1(行方向)


# ＜目次＞
# 0 準備
# 1 データ基準化


# 0 準備 ---------------------------------------------------------------------------------------------

# ライブラリ
from sklearn.datasets import load_iris
from sklearn.preprocessing import scale


# データロード
iris = load_iris()

# データ確認
iris.data[1:5, :]


# 1 データ基準化 ---------------------------------------------------------------------------------------

# インスタンス作成
iris_scaled = scale(iris.data, axis=0)

# 確認
iris_scaled.mean(axis=0)
iris_scaled.std(axis=0)
