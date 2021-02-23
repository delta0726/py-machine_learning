# ******************************************************************************
# Chapter   : 7 交差検証とモデル構築後のフロー
# Title     : 7-7 scikit-learnによるグリッドサーチ（Recipe56)
# Created by: Owner
# Created on: 2020/12/28
# Page      : P212 - P214
# ******************************************************************************

# ＜概要＞
# - scikit-learnのGridSearchCVを使ってirisにおける最適な最近傍モデルを選択する


# ＜目次＞
# 0 準備


# 0 準備 ------------------------------------------------------------------------------------------

from sklearn import datasets

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# データ準備
iris = datasets.load_iris()

# データ格納
X = iris.data[:, 2:]
y = iris.target

# データ分割
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, stratify=y, random_state=7)


# 1 モデリング -------------------------------------------------------------------------

# インスタンス生成
# --- k近傍法の分類器
knn_clf = KNeighborsClassifier()
vars(knn_clf)

# インスタンス生成
# --- グリッドサーチ
param_grid = {'n_neighbors': list(range(3, 9, 1))}
gs = GridSearchCV(knn_clf, param_grid=param_grid, cv=10)
vars(gs)

# 学習
# --- Holdout法
gs.fit(X_train, y_train)
vars(gs)

# 最良パラメータ
# --- Holdout法
gs.best_params_


# 2 クロスバリデーション ----------------------------------------------------------------

#
params = gs.cv_results_['params']
means = gs.cv_results_['mean_test_score']

for params, mean in zip(params, means):
    print(params, mean)


all_scores = []
for n_neighbors in range(3,9,1):
    knn_clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    all_scores.append((n_neighbors, cross_val_score(knn_clf, X_train, y_train, cv=10).mean()))
sorted(all_scores, key=lambda x: x[1], reverse=True)
