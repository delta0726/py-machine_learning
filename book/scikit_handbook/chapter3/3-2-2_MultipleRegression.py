# ******************************************************************************
# Title     : Scikit-Learnデータ分析実装ハンドブック
# Chapter   : 3 回帰
# Theme     : 3-2 重回帰による住宅価格予測モデルの実装
# Created by: Owner
# Created on: 2021/5/14
# Page      : P76 - P84
# ******************************************************************************


# ＜概要＞
# - 機械学習の基本フローの確認
#   --- 外部からデータをインポート
#   --- 決定木によるマルチクラス問題


# ＜目次＞
# 0 準備
# 1 モデルデータの作成
# 2 データ分割
# 3 前処理
# 4 予測
# 5 モデル精度の評価
# 6 残差プロットの作成


# 0 準備 -------------------------------------------------------------------------------------

# ライブラリ
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# データロード
boston = load_boston()

# データフレーム作成
df = pd.DataFrame(boston.data, columns=boston.feature_names).assign(MEDV=boston.target)

# データ確認
df.head()
df.shape


# 1 モデルデータの作成 ------------------------------------------------------------------

# データ格納
X = df.iloc[:, 0:13].values
y = df['MEDV'].values

# データ確認
X[:3]
y[:3]


# 2 データ分割 -------------------------------------------------------------------------

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# データ確認
X_train.shape
X_test.shape
y_train.shape
y_test.shape


# 3 前処理 ----------------------------------------------------------------------------

# インスタンス生成
# ---特徴量の標準化
sc = StandardScaler()

# データ変換
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

# データ確認
X_train[0]
X_train_std[0]


# 3 学習 ------------------------------------------------------------------------------------

# データ確認
# --- 特徴量は13個
X_train_std.shape

# インスタンス作成
# --- 線形回帰
model = LinearRegression()

# 学習
model.fit(X_train_std, y_train)

# 確認
pprint(vars(model))

# パラメータの取り出し
# --- 傾き（特徴量の重要度）
# --- 切片
model.coef_
model.intercept_


# 4 予測 -------------------------------------------------------------------------------------

# 予測データの作成
# --- 訓練データ
# --- テストデータ
y_train_pred = model.predict(X_train_std)
y_test_pred = model.predict(X_test_std)


# 5 モデル精度の評価 -----------------------------------------------------------------------------

# ＜ポイント＞
# - MSEが小さいほど誤差が少なく、モデル精度が高い
# - テストデータで評価することでモデルの客観性を担保する


# MSEの計算
# --- 手動計算
np.mean((y_train - y_train_pred) ** 2)
np.mean((y_test - y_test_pred) ** 2)

# MSEの計算
# --- {sklearn}の関数を用いる
mean_squared_error(y_true=y_train, y_pred=y_train_pred)
mean_squared_error(y_true=y_test, y_pred=y_test_pred)


# 6 残差プロットの作成 ----------------------------------------------------------------------------

#プロットのサイズ指定
plt.figure(figsize=(8,4))

# プロット作成
plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='red', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='blue', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.tight_layout()
plt.show()
