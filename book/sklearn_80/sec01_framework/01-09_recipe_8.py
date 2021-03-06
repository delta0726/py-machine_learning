# ******************************************************************************
# Chapter   : 1 機械学習の枠組みを理解する
# Title     : 1.9 すべてを1つにまとめる(Recipe8)
# Created by: Owner
# Created on: 2020/12/22
# Page      : P30 - P31
# ******************************************************************************


# ＜概要＞
# - 機械学習プロセスをk近傍法で確認する


# ＜目次＞
# 0 準備
# 1 データ分割
# 2 パラメータごとの予測精度
# 3 パラメータチューニングの基礎


# 0 準備 -------------------------------------------------------------------------------------------

# ライブラリ
import sklearn

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


# ライブラリ構成
dir(sklearn.model_selection)
dir(sklearn.neighbors)
dir(sklearn.model_selection)


# データ準備
iris = datasets.load_iris()
data = iris.data
target = iris.target

# 系列準備
x = iris.data[:, :2]
y = iris.target


# 1 データ分割 ------------------------------------------------------------------------------------

# データ分割
# --- 層化サンプリング
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=0)


# 2 パラメータごとの予測精度 -----------------------------------------------------------------------

# インスタンス生成
# --- 学習器の作成
knn_3_clf = KNeighborsClassifier(n_neighbors=3)
knn_5_clf = KNeighborsClassifier(n_neighbors=5)

# クロスバリデーション
knn_3_scores = cross_val_score(knn_3_clf, x_train, y_train, cv=10)
knn_5_scores = cross_val_score(knn_5_clf, x_train, y_train, cv=10)

# モデル評価
# --- ｢n_neighbors=5｣にすることで予測精度が改善
print("knn_3_score mean:", round(knn_3_scores.mean(), 3), "std:", round(knn_3_scores.std(), 3))
print("knn_5_score mean:", round(knn_5_scores.mean(), 3), "std:", round(knn_5_scores.std(), 3))


# 3 パラメータチューニングの基礎 ------------------------------------------------------------------

# リスト準備
all_scores = []

# チューニング
# --- パラメータごとのバリデーションスコア平均を格納
for n_neighbors in range(3, 9, 1):
    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    all_scores.append((n_neighbors,
                       cross_val_score(knn_clf, x_train, y_train, cv=10).mean()))

# 確認
# --- n_neighbors=4が最も精度が高い
all_scores
