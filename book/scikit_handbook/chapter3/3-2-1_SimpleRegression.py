# ******************************************************************************
# Title     : Scikit-Learnデータ分析実装ハンドブック
# Chapter   : 3 回帰
# Theme     : 3-2-1 単回帰による住宅価格予測モデルの実装
# Created by: Owner
# Created on: 2021/5/14
# Page      : P66 - P72
# ******************************************************************************


# ＜概要＞
# - 単回帰モデルを線形回帰モデルで実行


# ＜目次＞
# 0 準備
# 1 学習用データの作成
# 2 アルゴリズムの選択
# 3 学習
# 4 予測
# 5 プロット用データの作成
# 6 プロット作成


# 0 準備 -------------------------------------------------------------------------------------

# ライブラリ
import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

# データロード
boston = load_boston()

# データフレーム作成
df = pd.DataFrame(boston.data, columns=boston.feature_names).assign(MEDV=boston.target)

# データ確認
df.head()
df.shape


# 1 学習用データの作成 ------------------------------------------------------------------

# ＜ポイント＞
# - 特徴量はX(大文字)、ラベルはy(小文字)で表す
#   --- Xは複数の特徴量を持つことを意味する
# - 今回はデータ分割は行わない

# データ格納
# --- 先頭の20レコードのみ抽出
# --- Xは2次元配列で管理
X = df[:20][['RM']].values
y = df[:20]['MEDV'].values

# データ形状
X.shape
y.shape


# 2 アルゴリズムの選択 ----------------------------------------------------------------------------

# インスタンス作成
# --- 線形回帰
model = LinearRegression()

# 確認
pprint.pprint(vars(model))


# 3 学習 ---------------------------------------------------------------------------------------

# 学習
model.fit(X, y)

# 確認
pprint.pprint(vars(model))

# 傾きと切片
model.coef_
model.intercept_


# 4 予測 ---------------------------------------------------------------------------------------

# 予測用データ
new_data = np.array([[6]])

# 予測
model.predict(X=new_data)


# 5 プロット用データの作成 --------------------------------------------------------------------------

# データ作成
# --- 1次元配列
X_plt = np.arange(5, 9, 1)
X_plt

# データ作成
# --- 2次元配列
X_plt = np.arange(5, 9, 1)[:, np.newaxis]

# 予測
y_pred = model.predict(X_plt)
y_pred


# 6 プロット作成 -----------------------------------------------------------------------------------

# プロットサイズの指定
plt.figure(figsize=(8, 4))

# プロット作成
plt.scatter(X, y, color='blue', label='data')
plt.plot(X_plt, y_pred, color='red', linestyle='-', label='LinearRegression')
plt.ylabel('Price in $1000s [MEDV]')
plt.xlabel('Average number of rooms [RM]')
plt.title('Boston house-prices')
plt.legend(loc='lower right')
plt.show()
