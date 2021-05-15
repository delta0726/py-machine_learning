# ******************************************************************************
# Title     : Scikit-Learnデータ分析実装ハンドブック
# Chapter   : 3 回帰
# Theme     : 3-2-5 多項式回帰による住宅価格予測モデルの実装
# Created by: Owner
# Created on: 2021/5/15
# Page      : P89 - P93
# ******************************************************************************


# ＜概要＞
# - 特徴量を2次まで拡張して、重回帰と比べてどの程度MSEが改善するかを確認する


# ＜ポイント＞
# - 多項式変換は前処理の一環として有力な手段となりえる


# ＜目次＞
# 0 準備
# 1 データ分割
# 2 特徴量の多項式変換
# 3 データ基準化
# 4 モデル構築
# 5 モデル評価


# 0 準備 -------------------------------------------------------------------------------------

# ライブラリ
from pprint import pprint

import pandas as pd
from sklearn.datasets import load_boston
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


# 元データの確認
# --- 特徴量は13個
X_train.shape
X_test.shape

# インスタンス生成
POLY = PolynomialFeatures(degree=2, include_bias=False)

# データ変換
X_train_pol = POLY.fit_transform(X_train)
X_test_pol = POLY.fit_transform(X_test)

# データ確認
# --- 特徴量が104個に増えている
X_train_pol.shape
X_test_pol.shape

# 参考：変換器の確認
X_train_check = POLY.fit(X_train)
pprint(vars(X_train_check))


# 3 データ基準化 ---------------------------------------------------------------------------------

# インスタンス生成
# --- 標準化
sc = StandardScaler()

# 訓練データの標準化
X_train_std = sc.fit_transform(X_train_pol)
X_test_std = sc.fit_transform(X_test_pol)


# 4 モデル構築 -------------------------------------------------------------------------------------

# インスタンス生成
# --- 線形回帰モデルの構築
model = LinearRegression()

# モデル訓練
model.fit(X_train_std, y_train)

# 確認
pprint(vars(model))


# 5 モデル評価 ------------------------------------------------------------------------------------

# 予測データの作成
y_train_pred = model.predict(X_train_std)
y_test_pred = model.predict(X_test_std)

# MSEの計算
# --- 訓練データのMSEは下がったが、テストデータは同程度（過学習）
# --- MSE_train: 4.34
# --- MSE_test : 37.38
mean_squared_error(y_true=y_train, y_pred=y_train_pred)
mean_squared_error(y_true=y_test, y_pred=y_test_pred)
