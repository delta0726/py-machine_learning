# ******************************************************************************
# Chapter   : 9 決定木アルゴリズムとアンサンブル学習
# Title     : 9-2 決定木を使って基本的な分類を行う（Recipe69)
# Created by: Owner
# Created on: 2020/12/28
# Page      : P258 - P259
# ******************************************************************************

# ＜概要＞
# - ツリー系学習器の基本である決定木を確認する


# ＜目次＞
# 0 準備
# 1 決定木の実行


# 0 準備 ---------------------------------------------------------------------------

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# データ準備
iris = load_iris()

# データ格納
# --- yはSpeciesなので分類問題
X = iris.data
y = iris.target

# データ分割
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, stratify=y)


# 1 決定木の実行 --------------------------------------------------------------------

# インスタンス生成
dtc = DecisionTreeClassifier()
vars(dtc)

# 学習
dtc.fit(X_train, y_train)
vars(dtc)

# 予測
y_pred = dtc.predict(X_test)

# 予測精度の検証
accuracy_score(y_test, y_pred)
