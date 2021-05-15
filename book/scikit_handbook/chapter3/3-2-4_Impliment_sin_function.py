# ******************************************************************************
# Title     : Scikit-Learnデータ分析実装ハンドブック
# Chapter   : 3 回帰
# Theme     : 3-2-4 多項式回帰によるsin関数の実装
# Created by: Owner
# Created on: 2021/5/15
# Page      : P86 - P89
# ******************************************************************************


# ＜概要＞
# - 多項式の使用例としてsin関数を実装する


# ＜目次＞
# 0 準備
# 1 データ準備
# 2 多項式変換による前処理
# 3 学習
# 4 プロット


# 0 準備 -------------------------------------------------------------------------------------

# ライブラリ
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
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


# 2 多項式変換による前処理 --------------------------------------------------------------

# ＜ポイント＞
# - 特徴量が1次の項だとsin関数が表現できないので、3次多項式に変換する

# インスタンス生成
# --- 多項式回帰モデル
POLY = PolynomialFeatures(degree=3, include_bias=False)

# データ変換
X_pol = POLY.fit_transform(X)

# データ確認
# --- 3次に変換されている
X
X_pol


# 3 学習 -------------------------------------------------------------------------

# インスタンス生成
# --- 線形回帰モデル
model = LinearRegression()

# 学習
# --- 多項式変換した特徴量と正解で学習
model.fit(X_pol, y)

# データ確認
pprint(vars(model))


# 4 プロット -------------------------------------------------------------------------------

# データ作成
# --- X: プロット用にデータX_pltを作成
# --- y: 正解のプロット
X_plt = np.arange(0, 4, 0.1)[:, np.newaxis]
y_true = np.sin(1/4 * 2 * np.pi * X_plt ).ravel()

# 予測データの作成
y_pred = model.predict(POLY.transform(X_plt))


#プロットのサイズ指定
plt.figure(figsize=(8,4))

# sin関数の線形回帰によるモデル化
plt.scatter(X, y, color='blue', label='data')
plt.plot(X_plt, y_true, color='lime', linestyle='-', label='True sin(X)')
plt.plot(X_plt, y_pred, color='red', linestyle='-', label='LinearRegression  (degree=3)')
plt.legend(loc='upper right')

plt.show()
