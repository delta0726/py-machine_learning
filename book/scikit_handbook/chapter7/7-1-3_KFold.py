# ******************************************************************************
# Title     : Scikit-Learnデータ分析実装ハンドブック
# Chapter   : 7 モデル評価
# Theme     : 7-3 KFold
# Created by: Owner
# Created on: 2021/5/21
# Page      : P286 - P287
# ******************************************************************************


# ＜概要＞
# - 分類問題のクロスバリデーションは層化サンプリングがデフォルトで適用される
#   --- cross_val_score()の機能
#   --- KFoldクラスのshuffle引数をTrueにすることで回避することが可能


# ＜目次＞
# 0 準備
# 1 層化k分割交差検証
# 2 層化なしk分割交差検証


# 0 準備 ------------------------------------------------------------------------------

# ライブラリ
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# データロード
iris = load_iris()

# データ格納
X = iris.data
y = iris.target


# 1 層化k分割交差検証 ---------------------------------------------------------------------

# ＜ポイント＞
# - 分類問題の場合は、cross_val_score()はデフォルトで層化サンプリングが行われる
#   --- Foldごとのラベル割合が等しくなる


# インスタンス生成
# --- モデル
model = LogisticRegression()

# Fold数の決定
# --- 分類はデフォルトでは層化交差検証だが、Foldを追加うと回帰でデフォルトの交差検証となる
kfold = KFold(n_splits=3)

# モデル評価
# --- クロスバリデーションで分割したFoldごとのテストデータでモデル精度を算出
# --- データ分割の関数ではない点に注意
scores = cross_val_score(model, X, y, cv=kfold)

# データ確認
# --- 同じ値となる
scores


# 2 層化なしk分割交差検証 -----------------------------------------------------------------

# ＜ポイント＞
# - KFoldオブジェクトを使うと、分類問題でも層化サンプリングが行われないようにすることが可能
#   --- KFoldのsuffle引数をTrueとする


# インスタンス生成
# --- モデル
model = LogisticRegression()

# Fold数の決定
# --- 分類はデフォルトでは層化交差検証だが、Foldを追加うと回帰でデフォルトの交差検証となる
kfold = KFold(n_splits=3, shuffle=True, random_state=0)

# モデル評価
# --- クロスバリデーションで分割したFoldごとのテストデータでモデル精度を算出
# --- データ分割の関数ではない点に注意
scores = cross_val_score(model, X, y, cv=kfold)

# データ確認
# --- 値が異なる
scores
