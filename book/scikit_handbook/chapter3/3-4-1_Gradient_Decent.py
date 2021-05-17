# ******************************************************************************
# Title     : Scikit-Learnデータ分析実装ハンドブック
# Chapter   : 3 回帰
# Theme     : 3-4-1 確率的勾配降下法回帰による住宅価格予測モデルの実装
# Created by: Owner
# Created on: 2021/5/16
# Page      : P115 - P118
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
# 3 確率的勾配降下法による学習
# 4 モデル評価


# 0 準備 -------------------------------------------------------------------------------------

# ライブラリ
from pprint import pprint

import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
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


# 1 データ分割 --------------------------------------------------------------------------------

# データ格納
X = boston.data
y = boston.target

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# 2 データ基準化 ---------------------------------------------------------------------------------

# インスタンス生成
# --- 標準化
sc = StandardScaler()

# 特徴量の標準化
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)


# 3 確率的勾配降下法による学習 --------------------------------------------------------------------

# モデル構築
# --- 正則化を無効にするためalphaは微小な値を設定
model = SGDRegressor(loss='squared_loss', max_iter=100, eta0=0.01, learning_rate='constant',
                     alpha=1e-09, penalty='l2', l1_ratio=0, random_state=0)

# モデル訓練
model.fit(X_train_std, y_train)

# 確認
pprint(vars(model))


# 4 モデル評価 ----------------------------------------------------------------------------------

# 予測データの作成
y_train_pred = model.predict(X_train_std)
y_test_pred = model.predict(X_test_std)

# MSEの計算
# --- MSE_train: 19.54
# --- MSE_test : 35.67
mean_squared_error(y_train, y_train_pred)
mean_squared_error(y_test, y_test_pred)
