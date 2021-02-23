# ******************************************************************************
# Chapter   : 9 決定木アルゴリズムとアンサンブル学習
# Title     : 9-10 AdaBoost回帰器のチューニング（Recipe77)
# Created by: Owner
# Created on: 2020/12/30
# Page      : P295 - P298
# ******************************************************************************

# ＜概要＞
# - Adaboost回帰の重要なパラメータはlearning_rate(学習率)とLoss(損失関数)


# ＜目次＞
# 0 準備
# 1 層化サンプリング
# 2 初回チューニング
# 3 詳細チューニング
# 4 モデリング
# 5 モデル評価


# 0 準備 ------------------------------------------------------------------------------------------

import copy
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error


# データロード
cali_housing = fetch_california_housing()

# データ格納
X = cali_housing.data
y = cali_housing.target


# 1 層化サンプリング -----------------------------------------------------------------------------

# 階層サンプリングのため離散化
bins = np.arange(6)
binned_y = np.digitize(y, bins)

# データ分割
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, stratify=binned_y)


# 2 初回チューニング -------------------------------------------------------------------------------

# ＜ポイント＞
# - パラメータをざっくりどのあたりに設定すべきかを確認


# グリッド作成
# --- グリッドパターンを辞書に列挙
params_dist = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
    'loss': ['linear', 'square', 'exponential'],
}

# インスタンス生成
# --- ランダムサーチ・クロスバリデーション
# --- ベース学習器のインスタンスは引数内で設定
pre_gs_inst = RandomizedSearchCV(estimator=AdaBoostRegressor(),
                                 param_distributions=params_dist,
                                 cv=3, n_iter=10, n_jobs=-1)

# パラメータのチューニング
# --- ランダムサーチ
pre_gs_inst.fit(X_train, y_train)
vars(pre_gs_inst)

# 最良パラメータ
pre_gs_inst.best_params_


# 3 詳細チューニング ---------------------------------------------------------------------------

# ＜ポイント＞
# - モデル精度への影響が大きいパラメータをチューニング


# グリッド作成
# --- 既に決定したパラメータは固定する
params_dist = {
    'n_estimators': [100],
    'learning_rate': [0.04, 0.045, 0.05, 0.055, 0.06],
    'loss': ['linear'],
}

# パラメータのコピー
# --- 初回チューニングの最良モデル
ada_best = copy.deepcopy(pre_gs_inst.best_params_)
ada_best['n_estimators'] = 3000


# 4 モデリング ---------------------------------------------------------------------------

# モデル構築
rs_ada = AdaBoostRegressor(**ada_best)
vars(rs_ada)

# 学習
rs_ada.fit(X_train, y_train)


# 5 モデル評価 ---------------------------------------------------------------------------

# 予測
y_pred = rs_ada.predict(X_test)

# 決定係数(R2)
r2_score(y_true=y_test, y_pred=y_pred)

# 平均二乗誤差(R2)
mean_absolute_error(y_true=y_test, y_pred=y_pred)

# 平均二乗誤差率(MAPE)
(np.abs(y_test - y_pred) / y_test).mean()
