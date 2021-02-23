# ******************************************************************************
# Chapter   : 2 モデル構築前のワークフローと前処理
# Title     : 2-7 外れ値が存在する状況での線形モデル（Recipe15)
# Created by: Owner
# Created on: 2020/12/24
# Page      : P61 - P66
# ******************************************************************************

# ＜概要＞
# - 線形回帰は異常値に弱い一方、ロバスト回帰のように頑健な推定を行う学習器もある


# ＜目次＞
# 0 準備
# 1 異常値を含む回帰
# 2 推定器のロバスト性の比較


# 0 準備 ------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, TheilSenRegressor, Ridge, \
                                RANSACRegressor, ElasticNet, HuberRegressor
from sklearn.metrics import r2_score, mean_absolute_error


# データ作成
# --- 傾きが2の直線に対応するデータ
num_points = 100
x_vals = np.arange(num_points)
y_truth = 2 * x_vals

# プロット作成
plt.plot(x_vals, y_truth)
plt.show()

# ノイズ追加
y_noisy = y_truth.copy()
y_noisy[20:40] = y_noisy[20:40] * (-4 * x_vals[20:40]) - 100

# プロット作成
plt.scatter(x_vals, y_noisy, marker='x')
plt.title("Noise in y-direction")
plt.xlim(0, 100)
plt.show()


# 1 異常値を含む回帰 ----------------------------------------------------------------------------

# ＜ポイント＞
# - 線形回帰(OLS)は異常値に弱いアルゴリズム
#   --- ロバスト回帰を用いれば堅牢な予測が可能となる（TheilSenRegressor）


# インスタンス生成
# --- リストに格納
# --- 個々のインスタンスはタプルに格納
named_estimators = [('OLS', LinearRegression()),
                    ('TSR', TheilSenRegressor())]

# 回帰直線の比較
# --- est[0]: 名前
# --- est[1]: インスタンス
# --- インスタンスから予測までをメソッドチェーンで記述
for num_index, est in enumerate(named_estimators):
    y_pred = est[1].fit(x_vals.reshape(-1, 1), y_noisy).predict(x_vals.reshape(-1, 1))
    print(est[0], "R2 : ", r2_score(y_truth, y_pred))
    print(est[0], "MAE: ", mean_absolute_error(y_truth, y_pred))
    plt.plot(x_vals, y_pred, label=est[0])
plt.plot(x_vals, y_truth, label='True_Line')
plt.legend(loc='upper left')
plt.show()

# データセットと回帰直線をプロット
for num_index, est in enumerate(named_estimators):
    y_pred = est[1].fit(x_vals.reshape(-1, 1), y_noisy).predict(x_vals.reshape(-1, 1))
    plt.plot(x_vals, y_pred, label=est[0])
plt.legend(loc='upper left')
plt.title('Noise in y-direction')
plt.xlim([0, 100])
plt.scatter(x_vals, y_noisy, marker='x', color='red')
plt.show()


# 2 推定器のロバスト性の比較 ------------------------------------------------------------------------

# インスタンス生成
# --- リストに格納
# --- 個々のインスタンスはタプルに格納
named_estimators = [('OLS', LinearRegression()),
                    ('Ridge', Ridge()),
                    ('TSR', TheilSenRegressor()),
                    ('RANSAC', RANSACRegressor()),
                    ('ENet', ElasticNet()),
                    ('Huber', HuberRegressor())]

# 回帰直線の比較
# --- OLS / Ridge / ENet は異常値の影響を受けている
# --- TSR / RANSAC / Huber はロバスト推定ができている
for num_index, est in enumerate(named_estimators):
    y_pred = est[1].fit(x_vals.reshape(-1, 1), y_noisy).predict(x_vals.reshape(-1, 1))
    print(est[0], "R2 : ", r2_score(y_truth, y_pred))
    print(est[0], "MAE: ", mean_absolute_error(y_truth, y_pred))
