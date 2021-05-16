# ******************************************************************************
# Title     : Scikit-Learnデータ分析実装ハンドブック
# Chapter   : 3 回帰
# Theme     : 3-3-2 Lasso回帰による住宅価格予測モデルの実装
# Created by: Owner
# Created on: 2021/5/17
# Page      : P103 - P106
# ******************************************************************************


# ＜概要＞
# - Bostonデータセットを多項式変換したものを線形回帰モデルとLASSOモデルで学習する
#   --- Lassoモデルが過学習に対してロバストであることを確認する


# ＜目次＞
# 0 準備
# 1 データ分割
# 2 特徴量の多項式変換
# 3 データ基準化
# 4 モデル構築
# 5 モデル精度の評価


# 0 準備 -------------------------------------------------------------------------------------

# ライブラリ
from pprint import pprint

import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

# データロード
boston = load_boston()

# データフレーム作成
df = pd.DataFrame(boston.data, columns=boston.feature_names).assign(MEDV=boston.target)

# データ確認
df.head()
df.shape


# 1 データ分割 -------------------------------------------------------------------------------

# データ格納
# --- X: 特徴量（13個）
# --- Y: ラベルデータ
X = df.iloc[:, 0:13].values
y = df['MEDV'].values

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# 2 特徴量の多項式変換 -----------------------------------------------------------------------

# ＜ポイント＞
# - 特徴量の数は104個となる（15! / 13!2! - 1）

# インスタンス生成
POLY = PolynomialFeatures(degree=2, include_bias=False)

# データ変換
X_train_pol = POLY.fit_transform(X_train)
X_test_pol = POLY.fit_transform(X_test)


# 3 データ基準化 ---------------------------------------------------------------------------------

# インスタンス生成
# --- 標準化
sc = StandardScaler()

# 訓練データの標準化
X_train_std = sc.fit_transform(X_train_pol)
X_test_std = sc.fit_transform(X_test_pol)


# 4 モデル構築 ------------------------------------------------------------------------------

# インスタンス生成
# --- モデル1: 線形回帰モデル
# --- モデル2: LASSOモデル
model1 = LinearRegression()
model2 = Lasso(alpha=0.1)

# モデル訓練
model1.fit(X_train_std, y_train)
model2.fit(X_train_std, y_train)

# 確認
pprint(vars(model1))
pprint(vars(model2))

# パラメータ確認
# --- 線形回帰モデル
model1.intercept_
model1.coef_.shape
model1.coef_

# パラメータ確認
# --- LASSOモデル
# --- 切片の値は線形回帰モデルと一緒
model2.intercept_
model2.coef_.shape
model2.coef_


# 5 モデル精度の評価 --------------------------------------------------------------------------------

# ＜ポイント＞
# - Lassoが線形回帰モデルに対してロバストであることが確認できる


# 予測データの作成
y_train_pred1 = model1.predict(X_train_std)
y_train_pred2 = model2.predict(X_train_std)
y_test_pred1 = model1.predict(X_test_std)
y_test_pred2 = model2.predict(X_test_std)

# MSE
# --- Train: 4.34
# --- Test : 37.38
mean_squared_error(y_true=y_train, y_pred=y_train_pred1)
mean_squared_error(y_true=y_test, y_pred=y_test_pred1)

# MSE
# --- Train: 11.92
# --- Test : 25.40
mean_squared_error(y_true=y_train, y_pred=y_train_pred2)
mean_squared_error(y_true=y_test, y_pred=y_test_pred2)
