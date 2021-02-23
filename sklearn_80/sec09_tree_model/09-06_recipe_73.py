# ******************************************************************************
# Chapter   : 9 決定木アルゴリズムとアンサンブル学習
# Title     : 9-6 交差検証を使って過学習を抑制する（Recipe73)
# Created by: Owner
# Created on: 2020/12/30
# Page      : P277 - P279
# ******************************************************************************

# ＜概要＞
# - チューニングでパラメータを適切に調整することは過学習抑制のファーストステップ
#   --- 決定木のインスタンスのデフォルト設定は過学習気味になっている


# ＜目次＞
# 0 準備
# 1 層化サンプリング
# 2 チューニングによる過学習の抑制
# 3 モデル評価
# 4 決定木の可視化


# 0 準備 ------------------------------------------------------------------------------------------

import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# データロード
diabetes = load_diabetes()

# データ格納
X = diabetes.data
y = diabetes.target

# 特徴量のラベル
X_feature_names = ['age', 'gender', 'body mass index', 'average blood pressure',
                   'bl_0', 'bl_1', 'bl_2', 'bl_3', 'bl_4', 'bl_5']


# 1 層化サンプリング -----------------------------------------------------------------------------

# 階層サンプリングのため離散化
bins = 50 * np.arange(8)
binned_y = np.digitize(y, bins)

# データ分割
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, stratify=binned_y)


# 2 チューニングによる過学習の抑制 ----------------------------------------------------------------

# インスタンス生成
# --- 決定木回帰
dtr = DecisionTreeRegressor()

# インスタンス生成
# --- グリッドサーチ
gs_inst = GridSearchCV(dtr, param_grid={'max_depth': [3, 5, 7, 9, 20]}, cv=10)

# グリッドサーチの実行
gs_inst.fit(X_train, y_train)

# 最良モデルの確認
# --- max_depth = 3 が最も性能のよい決定木
gs_inst.best_estimator_


# 3 モデル評価 --------------------------------------------------------------------------------

# 予測
y_pred = gs_inst.predict(X_test)

# 平均絶対誤差(MAE)
mean_absolute_error(y_true=y_test, y_pred=y_pred)

# 平均絶対誤差率(MAPE)
(np.abs(y_test - y_pred) / y_test).mean()


# 4 決定木の可視化 ----------------------------------------------------------------------------

# 省略
