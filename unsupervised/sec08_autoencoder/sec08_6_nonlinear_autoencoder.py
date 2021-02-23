# ***************************************************************************************
# Book      : Pythonではじめる教師なし学習
# Chapter   : 8.オートエンコーダのハンズオン
# Title     : 8.6 非線形オートエンコーダ
# Created by: Owner
# Created on: 2020/1/30
# Page      : P184 - P186
# ***************************************************************************************


# ＜概要＞
# - 未完備オートエンコーダの活性化関数に非線形関数を適用する


# ＜目次＞
# 0 準備
# 1 データ準備
# 2 データ加工
# 3 分析用データの作成


# 0 準備 ----------------------------------------------------------------------

import os

import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split

# 共通関数
from sec08_autoencoder.common_function import anomalyScores
from sec08_autoencoder.common_function import plotResults

# 1 データ準備 ------------------------------------------------------------------

# データ取得
# --- クレジットカードデータ
current_path = os.getcwd()
file = os.path.sep.join(['', 'datasets', 'credit_card_data', 'credit_card.csv'])
data = pd.read_csv(current_path + file)

# データ格納
# --- コピーで別オブジェクトとして定義
dataX = data.copy().drop(['Class', 'Time'], axis=1)
dataY = data['Class'].copy()

# スケーリング
featuresToScale = dataX.columns
sX = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
dataX.loc[:, featuresToScale] = sX.fit_transform(dataX[featuresToScale])

# データ分割
X_train, X_test, y_train, y_test = \
    train_test_split(dataX, dataY, test_size=0.33,
                     random_state=2018, stratify=dataY)

# オブジェクト複製
X_train_AE = X_train.copy()
X_test_AE = X_test.copy()


# 2 準備 ----------------------------------------------------------------------

# 元のデータセットの次元数
# --- 29
X_train.shape

# モデル構築
# --- 活性化関数にReLuを使用
# --- 29 -> 27 -> 22 -> 27 -> 29
model = Sequential()
model.add(Dense(units=27, activation='relu', input_dim=29))
model.add(Dense(units=22, activation='relu'))
model.add(Dense(units=27, activation='relu'))
model.add(Dense(units=29, activation='relu'))

# コンパイル
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

# 学習
history = model.fit(x=X_train_AE, y=X_train_AE,
                    epochs=10,
                    batch_size=32,
                    shuffle=True,
                    validation_data=(X_train_AE, X_train_AE),
                    verbose=1)


# 予測
predictions = model.predict(X_test, verbose=1)

# モデル評価
# --- 10回のシミュレーションは省略
# --- 平均適合率は0.22
# --- 変動係数は0.06
anomalyScoresAE = anomalyScores(X_test, predictions)
preds = plotResults(y_test, anomalyScoresAE, True)

# 後処理
model.reset_states()
