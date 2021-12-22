# ******************************************************************************
# Title       : AI・データサイエンスのための数学プログラミング
# Chapter     : 2 機械学習を使った分析を行ってみよう
# Theme       : 2-10 サポートベクトル回帰によって時系列予測をしてみよう
# Creat Date  : 2021/12/23
# Final Update:
# Page        : P89 - P91
# ******************************************************************************


# ＜概要＞
# - 機械学習は分類に加えて回帰も行うことができる


# ＜目次＞
# 0 準備
# 1 回帰用のデータ作成
# 2 モデル構築
# 3 プロット作成


# 0 準備 ---------------------------------------------------------------------

# ライブラリ
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import svm


# データ読み込み
df_info = pd.read_csv("csv/accomodation_data.csv", index_col=0)


# 1 回帰用のデータ作成 -------------------------------------------------------

# データ作成
data_e = df_info.loc[:, lambda x: x.columns.str.startswith("X")]
label = df_info.loc[:, lambda x: x.columns.str.startswith("Y")]

# データを作成
data_target = data_e[label['Y'] == 1]
data_y = data_target
data_x = np.stack([np.arange(0, len(data_target[0])) for _ in range(len(data_target))], axis=0)
data_y = np.ravel(data_y)
data_x = np.ravel(data_x)

# データ分割
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y)


# 2 モデル構築 ------------------------------------------------------

# 訓練データによるモデル構築（サポートベクトル回帰）
model = svm.SVR(kernel='rbf', C=1)
reg = model.fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))

# 決定係数R^2
reg.score(x_test.reshape(-1, 1), y_test.reshape(-1, 1))


# 3 プロット作成 ------------------------------------------------------

# 予測曲線を描画
x_pred = np.arange(len(data_target[0])).reshape(-1, 1)
y_pred = model.predict(x_pred)
plt.plot(data_x, data_y, "k.")
plt.plot(x_pred, y_pred, "r.-")
plt.show()

