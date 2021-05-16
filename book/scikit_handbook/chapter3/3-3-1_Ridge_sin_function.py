# ******************************************************************************
# Title     : Scikit-Learnデータ分析実装ハンドブック
# Chapter   : 3 回帰
# Theme     : 3-3-1 Ridge回帰によるsin関数の実装
# Created by: Owner
# Created on: 2021/5/16
# Page      : P98 - P101
# ******************************************************************************


# ＜概要＞
# - 6次の特徴量を生成して線形回帰モデルで学習させることで、過学習がどのような状態かを確認する
#   --- Ridge回帰が過学習に対してロバストであることを確認


# ＜目次＞
# 0 準備
# 1 訓練データの作成
# 2 特徴量の多項式変換
# 3 モデル構築
# 4 プロット


# 0 準備 --------------------------------------------------------------------------------

from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures



# 1 訓練データの作成 ---------------------------------------------------------------------

# ＜ポイント＞
# - {sklearn}では、Xは2次元配列、yは1次元配列でインプットする
# - np.newaxisで1次元配列を2次元配列に変換
# - ravelメソッドで2次元配列を1次元配列に変換


# シード設定
np.random.seed(8)

# データ作成
# --- X: 0-4で一様乱数を15個作成
# --- Y: sin関数に正規乱数を加えたもの
X = np.random.uniform(0, 4, 15)[:, np.newaxis]
y = np.sin(1/4 * 2 * np.pi * X).ravel() + np.random.normal(0, 0.3, 15)


# 2 特徴量の多項式変換 -------------------------------------------------------------------

# 特徴量の多項式変換
POLY = PolynomialFeatures(degree=6, include_bias=False)
X_pol = POLY.fit_transform(X)

# 特徴量の数
# --- 特徴量から1次元から6次元に変換
X.shape
X_pol.shape


# 3 モデル構築 ---------------------------------------------------------------------------

# インスタンス生成
# --- 線形回帰モデルをベンチマークとして設定
# --- alpha: L2正則化の強さ
# --- alphaがゼロに近くなると線形回帰モデルに近づく
model1 = LinearRegression()
model2 = Ridge(alpha=0.1)
#model2 = Ridge(alpha=0.0001)

# モデル訓練
model1.fit(X_pol, y)
model2.fit(X_pol, y)


# 4 プロット -------------------------------------------------------------------------------

# データ作成
# --- X: プロット用にデータX_pltを作成
# --- y: 正解のプロット
X_plt = np.arange(0, 4, 0.1)[:, np.newaxis]
y_true = np.sin(1/4 * 2 * np.pi * X_plt).ravel()

# 予測データの作成
y_pred1 = model1.predict(POLY.transform(X_plt))
y_pred2 = model2.predict(POLY.transform(X_plt))

#プロットのサイズ指定
plt.figure(figsize=(8,4))

# sin関数の線形回帰によるモデル化
plt.scatter(X, y, color='blue', label='data')
plt.plot(X_plt, y_true, color='lime', linestyle='-', label='True sin(X)')
plt.plot(X_plt, y_pred1, color='red', linestyle='-', label='LinearRegression')
plt.plot(X_plt, y_pred2, color='blue', linestyle='-', label='RidegeRegression')
plt.legend(loc='upper right')

plt.show()