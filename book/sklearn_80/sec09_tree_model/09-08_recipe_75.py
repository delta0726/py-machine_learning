# ******************************************************************************
# Chapter   : 9 決定木アルゴリズムとアンサンブル学習
# Title     : 9-8 再近傍に基づくバギング回帰（Recipe75)
# Created by: Owner
# Created on: 2020/12/30
# Page      : P283 - P286
# ******************************************************************************

# ＜概要＞
# - バギングはベース推定器の複数のインスタンスで構成され、推定器は訓練データをランダムなサブセットにして学習する


# ＜目次＞
# 0 準備
# 1 層化サンプリング
# 2 チューニング
# 3 モデリング
# 4 モデル評価


# 0 準備 ------------------------------------------------------------------------------------------

import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error


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


# 2 チューニング -----------------------------------------------------------------------------

# ＜パラメータ＞
# - max_samples : ベース推定器の訓練時にXから抽出するサンプルの数
# - max_features: ベース推定器の訓練時にXから抽出する特徴量の数
# - oob_score   : モデル構築時にサンプリングされなかったサンプルを使用するかどうか
# - n_estimators: ベース推定器の数

# グリッド作成
# --- グリッドパターンを辞書に列挙
# --- 以下の場合は2^4パターンが生成される
param_dist = {
    'max_samples': [0.5, 1.0],
    'max_features': [0.5, 1.0],
    'oob_score': [True, False],
    'base_estimator__n_neighbors': [3, 5],
    'n_estimators': [100]
}

# インスタンス生成
# --- ベース推定器
single_estimator = KNeighborsRegressor()
vars(single_estimator)

# インスタンス生成
# --- アンサンブル推定器
ensemble_estimator = BaggingRegressor(base_estimator=single_estimator)
vars(ensemble_estimator)

# インスタンス生成
# --- ランダムサーチ・クロスバリデーション
pre_gs_inst_bag = RandomizedSearchCV(ensemble_estimator,
                                     param_distributions=param_dist,
                                     cv=3, n_iter=5, n_jobs=-1)

# チューニング実行
pre_gs_inst_bag.fit(X_train, y_train)

# 最良パラメータの確認
pre_gs_inst_bag.best_params_


# 3 モデリング ---------------------------------------------------------------------------

# インスタンス生成
# --- チューニング結果をもとに設定
# --- 推定器の数を1000に増やす
rs_bag = BaggingRegressor(**{
    'max_features': 1.0,
    'max_samples': 0.5,
    'n_estimators': 1000,
    'oob_score': True,
    'base_estimator': KNeighborsRegressor(n_neighbors=5)
})

# 学習
rs_bag.fit(X_train, y_train)


# 4 モデル評価 ---------------------------------------------------------------------------

# 予測
y_pred = rs_bag.predict(X_test)

# 決定係数(R2)
r2_score(y_true=y_test, y_pred=y_pred)

# 平均二乗誤差(R2)
mean_absolute_error(y_true=y_test, y_pred=y_pred)

# 平均二乗誤差率(MAPE)
(np.abs(y_test - y_pred) / y_test).mean()
