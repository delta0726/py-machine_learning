# ******************************************************************************
# Chapter   : 7 交差検証とモデル構築後のフロー
# Title     : 7-2 交差検証を使ってモデルを選択する（Recipe51)
# Created by: Owner
# Created on: 2020/12/27
# Page      : P201 - P204
# ******************************************************************************

# ＜概要＞
# - クロスバリデーションを用いて最適なモデルを選択する
#   --- 最適なモデルとは、テストデータにおいて最も高いスコアを出すモデル


# ＜クロスバリデーションの流れ＞
# 1 fold2-4を用いてknn学習を行って学習器を作成する
# 2 fold1のデータの予測値を作成して、fold1の元データと評価する
# 3 1と2の操作を繰り返すことで、全データを1回はテストデータとして使用する


# ＜目次＞
# 0 準備
# 1 モデル構築
# 2 モデル評価


# 0 準備 ------------------------------------------------------------------------------------------

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


# データロード
iris = datasets.load_iris()

# データ格納
X = iris.data[:, 2:]
y = iris.target


# 1 モデル構築 -----------------------------------------------------------------------------

# データ分割
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, stratify=y, random_state=7)

# インスタンスの生成
kn_3 = KNeighborsClassifier(n_neighbors=3)
kn_5 = KNeighborsClassifier(n_neighbors=5)


# 2 モデル評価 -----------------------------------------------------------------------------

# クロスバリデーション
# --- 訓練データのみインプット
kn_3_scores = cross_val_score(kn_3, X_train, y_train, cv=4)
kn_5_scores = cross_val_score(kn_5, X_train, y_train, cv=4)

# 確認
# --- CVスコア
kn_3_scores
kn_5_scores

# CVスコアの平均
print('Mean of kn_3', kn_3_scores.mean())
print('Mean of kn_5', kn_5_scores.mean())

# CVスコアの標準偏差
print('Std of kn_3', kn_3_scores.std())
print('Std of kn_5', kn_5_scores.std())
