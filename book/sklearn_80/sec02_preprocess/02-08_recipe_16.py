# ******************************************************************************
# Chapter   : 2 モデル構築前のワークフローと前処理
# Title     : 2-8 パイプラインを使って全てを1つにまとめる（Recipe16)
# Created by: Owner
# Created on: 2020/12/24
# Page      : P66 - P69
# ******************************************************************************

# ＜概要＞
# - 前処理のステップをパイプラインで一元化する


# ＜目次＞
# 0 準備
# 1 パイプラインの実例(PCA)
# 2 パイプラインの仕組み
# 3 パラメータの設定


# 0 準備 ------------------------------------------------------------------------------------------

import numpy as np

from sklearn.datasets import load_iris
from sklearn import pipeline, decomposition
from sklearn.impute import SimpleImputer

# データ準備
iris = load_iris()
iris_data = iris.data

# 欠損値の挿入
mask = np.random.binomial(n=1, p=0.25, size=iris_data.shape).astype(bool)
iris_data[mask] = np.nan
iris_data[:5]


# 1 パイプラインの実例(PCA) ---------------------------------------------------------------------------

# インスタンス作成
pca = decomposition.PCA()
imputer = SimpleImputer(strategy='mean')

# パイプラインの作成
# --- パラメータ設定はしていない
pipe = pipeline.Pipeline([('imputer', imputer),
                          ('pca', pca)])

# 処理の適用
# --- PCAは初期設定では要素数と同じ数のPCを返す
iris_data_transformed = pipe.fit_transform(iris_data)
iris_data_transformed[:5]


# 2 パイプラインの仕組み -----------------------------------------------------------------------------

# ＜ポイント＞
# - パイプラインの各ステップは、タプルからなるリストを通じてパイプラインオブジェクトに渡される
#   --- タプルの要素は、1つ目が｢名前｣、2つ目が｢オブジェクト｣
#   --- make_pipeline()を使うと簡単にパイプラインオブジェクトを作ることができる

# パイプラインを簡単に構築
# --- make_pipeline()
# --- タプルとリストからなるパイプラインオブジェクト自体を作成する
pipe2 = pipeline.make_pipeline(imputer, pca)
pipe2.steps

# 処理の適用
iris_data_transformed2 = pipe2.fit_transform(iris_data)
iris_data_transformed2[:5]


# 3 パラメータの設定 -----------------------------------------------------------------------------

# パラメータ設定
# --- パイプラインのオブジェクトにパラメータを渡す
pipe2.set_params(pca__n_components=2)
pipe2.steps

# 処理の適用
iris_data_transformed3 = pipe2.fit_transform(iris_data)
iris_data_transformed3[:5]
