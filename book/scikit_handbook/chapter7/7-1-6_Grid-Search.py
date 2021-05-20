# ******************************************************************************
# Title     : Scikit-Learnデータ分析実装ハンドブック
# Chapter   : 7 モデル評価
# Theme     : 7-6 交差検証を用いたグリッドサーチ
# Created by: Owner
# Created on: 2021/5/21
# Page      : P289 - P290
# ******************************************************************************


# ＜概要＞
# - クロスバリデーションの評価手法を用いながらグリッドサーチによるパラメータチューニングを行う


# ＜目次＞
# 0 準備
# 1 グリッド設定
# 2 グリッドサーチ


# 0 準備 ------------------------------------------------------------------------------

# ライブラリ
from pprint import pprint

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# データロード
iris = load_iris()

# データ格納
X = iris.data
y = iris.target


# 1 グリッド設定 ----------------------------------------------------------------------

# グリッド設定
# --- サポートベクターのパラメータ
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
}

# グリッド作成
# --- インスタンス生成
grid_search = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=5)

# 確認
pprint(vars(grid_search))


# 2 グリッドサーチ --------------------------------------------------------------------

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# グリッドサーチの実行
grid_search.fit(X_train, y_train)

# 確認
pprint(vars(grid_search))

# スコア計算
grid_search.score(X_test, y_test)

# 結果確認
# --- 最良スコア
# --- 最適パラメータ
grid_search.best_score_
grid_search.best_params_
