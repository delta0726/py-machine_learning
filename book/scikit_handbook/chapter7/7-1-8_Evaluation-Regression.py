# ******************************************************************************
# Title     : Scikit-Learnデータ分析実装ハンドブック
# Chapter   : 7 モデル評価
# Theme     : 7-8 回帰評価方法
# Created by: Owner
# Created on: 2021/5/22
# Page      : P292 - P293
# ******************************************************************************


# ＜概要＞
# - 回帰モデルの評価指標を確認する


# ＜目次＞
# 0 準備
# 1 誤差指標
# 2 評価指標


# 0 準備 ------------------------------------------------------------------------------

# ライブラリ
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_erro
from sklearn.metrics import r2_score
from math import sqrt


# 1 誤差指標 ---------------------------------------------------------------------------

# データ作成
y_true = [1.2, 2.5, 1.9, 1.4, 0.4, 0.8, 2.2, 1.1, 3.3, 0.5]
y_pred = [0.4, 1.2, 0.5, 2.5, 3.3, 2.1, 0.9, 1.8, 1.4, 2.1]

# 平均二乗誤差
mean_squared_error(y_true=y_true, y_pred=y_pred)

# 通常尺度に変更
sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))


# 2 評価指標 ----------------------------------------------------------------------------

# データ作成
X = np.array([1.2, 2.5, 1.9, 1.4, 0.4, 0.8, 2.2, 1.1, 3.3, 0.5])[:, np.newaxis]
y = [0.4, 1.2, 0.5, 2.5, 3.3, 2.1, 0.9, 1.8, 1.4, 2.1]

# モデル
model = LinearRegression()

# 学習
model.fit(X, y)

# モデル評価
# --- R2：学習時に計算される損失関数の値
model.score(X, y)

# 予測
y_pred = model.predict(X)

# モデル評価
# --- R2
r2_score(y_true=y, y_pred=y_pred)
