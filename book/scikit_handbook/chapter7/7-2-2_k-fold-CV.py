# ******************************************************************************
# Title     : Scikit-Learnデータ分析実装ハンドブック
# Chapter   : 7 モデル評価
# Theme     : 7-2 k-分割交差検証
# Created by: Owner
# Created on: 2021/5/20
# Page      : P285 - P286
# ******************************************************************************


# ＜概要＞
# - データセット全体を訓練データとテストデータに複数回に分割することで、全データを訓練/テストに使用する
# - ホールドアウト法のk倍の計算時間がかかるのがデメリット


# ＜関数＞
# - cross_val_score()
#   --- Foldごとのモデル精度の評価を出力する関数であり、データ分割のアウトプットは得られない


# ＜参考＞
# クロスバリデーション
# https://docs.pyq.jp/python/machine_learning/glossary/cross_validation.html


# ＜目次＞
# 0 準備
# 1 モデル精度の評価


# 0 準備 ------------------------------------------------------------------------------

# ライブラリ
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# データロード
iris = load_iris()

# データ格納
X = iris.data
y = iris.target


# 1 モデル精度の評価 ---------------------------------------------------------------------

# インスタンス生成
# --- モデル
model = LogisticRegression()

# モデル評価
# --- クロスバリデーションで分割したFoldごとのテストデータでモデル精度を算出
# --- データ分割の関数ではない点に注意
scores = cross_val_score(model, X, y, cv=3)

# データ確認
scores
