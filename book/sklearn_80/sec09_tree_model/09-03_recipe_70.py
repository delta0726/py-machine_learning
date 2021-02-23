# ******************************************************************************
# Chapter   : 9 決定木アルゴリズムとアンサンブル学習
# Title     : 9-3 pydotを使って決定木を可視化する（Recipe70)
# Created by: Owner
# Created on: 2020/12/28
# Page      : P259 - P262
# ******************************************************************************

# ＜概要＞
# - sklearn.tree.plot_treeが見当たらず


# ＜目次＞
# 0 準備


# 0 準備 ----------------------------------------------------------------------------

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
#from sklearn.tree import plot_tree


# データロード
iris = load_iris()

# データ格納
X = iris.data
y = iris.target

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# 1 モデリング -----------------------------------------------------------------------

# インスタンス生成
clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)

# 学習
clf.fit(X_train, y_train)
vars(clf)


# 2 ツリープロットの作成 --------------------------------------------------------------

# plt.figure(figsize=(15, 10))
# plot_tree(clf, feature_names=iris.feature_names, filled=True)
# plt.show()
