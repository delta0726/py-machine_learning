# ******************************************************************************
# Title     : Scikit-Learnデータ分析実装ハンドブック
# Chapter   : 3 回帰
# Theme     : 3-5-1 線形サポートベクトル回帰を使った部屋数と住宅価格の予測モデルの実装
# Created by: Owner
# Created on: 2021/5/16
# Page      : P124 - P130
# ******************************************************************************


# ＜概要＞
# - 線形回帰モデルは正規方程式を解析的に解くことで算出するが、以下のケースで計算が困難になるケースがある
#   --- 非線形モデルなど解析的に解を見つけることができない
#   --- 大規模データで計算量が急激に増える
# - 確率的勾配降下法は計算精度は劣るものの、上記の問題を解消して解を得ることができる


# ＜目次＞
# 0 準備
# 1 データ分割
# 2 データ基準化
# 3 モデル訓練＆学習
# 4 プロット作成
# 5 モデル精度の評価


# 0 準備 -------------------------------------------------------------------------------------

# ライブラリ
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston

# データロード
boston = load_boston()


# 1 データ分割 ----------------------------------------------------------------------------------

# データ格納
X = boston.data[:20, [5]]
y = boston.target[:20]

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# 2 データ基準化 ---------------------------------------------------------------------------------

# インスタンス生成
# --- 基準化
sc = StandardScaler()

# データ変換
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)


# 3 モデル訓練＆学習 ----------------------------------------------------------------------------

# ＜ハイパーパラメータ＞
# kernel  : 計算に使用するカーネル
# C       : サポートベクトル回帰と正則化項のバランス
# epsilon : 予測から上下に広がるチューブの縦幅


# インスタンス生成
# --- 線形回帰モデル
# --- サポートベクトル回帰
# --- Cを大きくして正則化項のウエイトを下げる
model1 = LinearRegression()
model2 = SVR(kernel='linear', C=10000, epsilon=4)

# モデル訓練
model1.fit(X_train_std, y_train)
model2.fit(X_train_std, y_train)

# 確認
pprint(vars(model2))

# パラメータ確認
# --- サポートベクトルの特徴量
# --- サポートベクトルのインデックス
model2.support_vectors_
model2.support_


# 4 プロット作成 --------------------------------------------------------------------------------

#プロットのサイズ指定
plt.figure(figsize=(8, 4))

# 訓練データの最小値から最大値まで0.1刻みのX_pltを作成
X_plt = np.arange(X_train_std.min(), X_train_std.max(), 0.1)[:, np.newaxis]

# プロット作成
# --- 線形回帰
# --- SVR
y_plt_pred1 = model1.predict(X_plt)
y_plt_pred2 = model2.predict(X_plt)

# 部屋数と住宅価格の散布図とプロット
plt.scatter(X_train_std, y_train, color='blue', label='data')
plt.plot(X_plt, y_plt_pred1, color='lime', linestyle='-', label='LinearRegression')
plt.plot(X_plt, y_plt_pred2 ,color='red', linestyle='-', label='SVR')
plt.plot(X_plt, y_plt_pred2 + model2.epsilon, color='red', linestyle=':', label='margin')
plt.plot(X_plt, y_plt_pred2 - model2.epsilon, color='red', linestyle=':')
plt.ylabel('Price in $1000s [MEDV]')
plt.xlabel('Average number of rooms [RM]')
plt.title('Boston house-prices')
plt.legend(loc='lower right')

plt.show()


# 5 モデル精度の評価 -----------------------------------------------------------------------------

# 予測データの作成
# --- 線形回帰モデル
y_train_pred1 = model1.predict(X_train_std)
y_test_pred1 = model1.predict(X_test_std)

# 予測データの作成
# --- サポートベクトル回帰
y_test_pred2 = model2.predict(X_test_std)
y_train_pred2 = model2.predict(X_train_std)


# モデル精度の評価
# --- 線形回帰モデル
mean_squared_error(y_true=y_train, y_pred=y_train_pred1)
mean_squared_error(y_true=y_test, y_pred=y_test_pred1)

# モデル精度の評価
# --- サポートベクトル回帰
mean_squared_error(y_true=y_train, y_pred=y_train_pred2)
mean_squared_error(y_true=y_test, y_pred=y_test_pred2)

