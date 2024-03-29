# ******************************************************************************
# Chapter   : 1 機械学習の枠組みを理解する
# Title     : 1.4 irisデータセットの可視化(Recipe3)
# Created by: Owner
# Created on: 2020/12/20
# Page      : P12 - P14
# ******************************************************************************


# ＜概要＞
# - irisデータセットの要素を確認する
#   --- Rのirisのように単純なデータフレームにはなっていない
#   --- 機械学習を意識してラベル/特徴量などに分かれている


# ＜目次＞
# 0 準備
# 1 データセットの確認


# 0 準備 -------------------------------------------------------------------------------------------

# ライブラリ
from sklearn import datasets


# データロード
iris = datasets.load_iris()

# クラス
type(iris)


# 1 データセットの確認 ---------------------------------------------------------------

# ＜ポイント＞
# - irisデータセットは教師あり分類問題に対応したデータセット
#   --- ラベルデータがカテゴリカル変数
#   --- ラベルは3つの要素で構成される（マルチクラス分類問題）

# 特徴量データ
# --- データ要素
# --- 列名
iris.data[1:5, :]
iris.feature_names
iris.data.shape

# ラベルデータ
# --- ラベル系列データ（ラベル番号）
# --- ラベルのカテゴリ
iris.target
iris.target_names
iris.target.shape
