# ******************************************************************************
# Chapter   : 9 決定木アルゴリズムとアンサンブル学習
# Title     : 9-9 勾配ブースティング決定木のチューニング（Recipe76)
# Created by: Owner
# Created on: 2020/12/30
# Page      : P287 - P295
# ******************************************************************************

# ＜概要＞
#


# ＜目次＞
# 0 準備
# 1 層化サンプリング
# 2 チューニング
# 3 チューニング結果の表示
# 4 チューニング: 2回目
# 5 モデリング
# 6 モデル評価
# 7 分類問題のアプローチ


# 0 準備 ------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score


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


# 2 チューニング --------------------------------------------------------------------------------

# パラメータリスト
# --- lsは最小二乗法、huberは最小二乗法と最小絶対偏差を組み合わせた方法
param_dist = {
    'max_features': ['log2', 1.0],
    'max_depth': [3, 5, 7, 10],
    'min_samples_leaf': [2, 3, 5, 10],
    'n_estimators': [50, 100],
    'learning_rate': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.3],
    'loss': ['ls', 'huber']
 }

# インスタンス生成
pre_gs_inst = RandomizedSearchCV(GradientBoostingRegressor(warm_start=True),
                                 param_distributions=param_dist,
                                 cv=3, n_iter=30, n_jobs=-1)

# 確認
vars(pre_gs_inst)

# チューニング実行
pre_gs_inst.fit(X_train, y_train)


# 3 チューニング結果の表示 ----------------------------------------------------------------------------

# 関数定義
# --- 結果をデータフレームにまとめる
def get_grid_df(fitted_gs_estimator):
    res_dict = fitted_gs_estimator.cv_results_
    results_df = pd.DataFrame()
    for key in res_dict.keys():
        results_df[key] = res_dict[key]

    return results_df


# 関数定義
# --- データフレームをレポートとして出力する関数
def group_report(results_df):
    param_cols = [x for x in results_df.columns if 'param' in x and x is not 'params']

    print
    "Grid CV Report \n"

    output_df = pd.DataFrame(columns=['param_type', 'param_set', 'mean_score', 'mean_std'])
    cc = 0
    for param in param_cols:
        for key, group in results_df.groupby(param):
            output_df.loc[cc] = (param, key, group['mean_test_score'].mean(), group['mean_test_score'].std())
            cc += 1
    return output_df


# チューニング結果の出力
result_df = get_grid_df(pre_gs_inst)
group_report(result_df)


# 4 チューニング: 2回目 --------------------------------------------------------------------------

# パラメータリスト
param_dist = {
    'max_features': ['sqrt', 0.5, 1.0],
    'max_depth': [2, 3, 4],
    'min_samples_leaf': [3, 4],
    'n_estimators': [50, 100],
    'learning_rate': [0.2, 0.25, 0.3, 0.4],
    'loss': ['ls', 'huber']
 }

# インスタンス生成
# --- ランダムサーチ・クロスバリデーション
pre_gs_inst = RandomizedSearchCV(GradientBoostingRegressor(warm_start=True),
                                 param_distributions=param_dist, cv=3, n_iter=30, n_jobs=-1)

# 学習
pre_gs_inst.fit(X_train, y_train)

# チューニング結果の出力
result_df = get_grid_df(pre_gs_inst)
group_report(result_df)


# 5 モデリング --------------------------------------------------------------------------

# パラメータリスト
param_dist = {
    'max_features': [0.4, 0.5, 0.6],
    'max_depth': [5, 6],
    'min_samples_leaf': [4, 5],
    'n_estimators': [300],
    'learning_rate': [0.3],
    'loss': ['ls', 'huber']
 }

# インスタンス生成
rs_gbt = GradientBoostingRegressor(warm_start=True,
                                   max_features=0.5,
                                   min_samples_leaf=4,
                                   learning_rate=0.3,
                                   max_depth=6,
                                   n_estimators=4000,
                                   loss='huber')

# 学習
pre_gs_inst.fit(X_train, y_train)


# 6 モデル評価 ---------------------------------------------------------------------------

# 予測
y_pred = pre_gs_inst.predict(X_test)

# 決定係数(R2)
r2_score(y_true=y_test, y_pred=y_pred)

# 平均二乗誤差(R2)
mean_absolute_error(y_true=y_test, y_pred=y_pred)

# 平均二乗誤差率(MAPE)
(np.abs(y_test - y_pred) / y_test).mean()


# 7 分類問題のアプローチ ------------------------------------------------------------------

# ラベルのヒストグラム
# --- 連続データ
pd.Series(y).hist(bins=50)
plt.show()

# 連続データの離散化
# --- ラベルデータ
bins = np.arange(6)
binned_y = np.digitize(y, bins=bins)

# データ分割
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, stratify=binned_y)

# 行数取得
# --- 訓練データ
train_shape = X_train.shape[0]

# ラベル変換
# --- ビン情報をもとにバイナリデータに変換
y_binary = np.where(y >= 5, 1, 0)
y_train_binned = y_binary[:train_shape]
y_test_binned = y_binary[train_shape:]

# パラメータリスト
param_dist = {
    'max_features': ['log2', 0.5, 1.0],
    'max_depth': [2, 3, 6],
    'min_samples_leaf': [1, 2, 3, 10],
    'n_estimators': [100],
    'learning_rate': [0.1, 0.2, 0.3, 1],
    'loss': ['deviance']
}

# インスタンス生成
# --- ランダムサーチ・クロスバリデーション
pre_gs_inst = RandomizedSearchCV(GradientBoostingClassifier(warm_start=True),
                                 param_distributions=param_dist,
                                 cv=3, n_iter=10, n_jobs=-1)

# チューニング
pre_gs_inst.fit(X_train, y_train_binned)

# ベストパラメータの取得
pre_gs_inst.best_params_

# インスタンス生成
# --- 勾配ブースティング分類器
gbc = GradientBoostingClassifier(**{
    'learning_rate': 0.2,
    'loss': 'deviance',
    'max_depth': 2,
    'max_features': 1.0,
    'min_samples_leaf': 2,
    'n_estimators': 1000,
    'warm_start': True}).fit(X_train, y_train_binned)

# 予測
y_pred = gbc.predict(X_test)

# モデル評価
accuracy_score(y_test_binned, y_pred)
