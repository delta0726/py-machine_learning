# ***************************************************************************************
# Book      : Pythonではじめる教師なし学習
# Chapter   : 8.オートエンコーダのハンズオン
# Title     : 8.11-12 ノイズ除去オートエンコーダ
# Created by: Owner
# Created on: 2020/2/5
# Page      : P194 - P202
# ***************************************************************************************


# ＜概要＞
# - 実世界では多くのノイズが発生するので、ノイズに対して頑健なオートエンコーダが求められる
# - 背後の構造は学習するが、ノイズは学習せずに適切に排除することが求められる


# ＜目次＞
# 0 準備
# 1 データ準備
# 2 ノイズデータの作成
# 3 ノイズ除去の未完備オートエンコーダ
# 4 ノイズ除去の過完備オートエンコーダ
# 5 Reluを用いたノイズ除去の過完備オートエンコーダ


# 0 準備 ----------------------------------------------------------------------

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras import regularizers
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split

# 共通関数
from sec08_autoencoder.common_function import anomalyScores
from sec08_autoencoder.common_function import plotResults


# 1 データ準備 ---------------------------------------------------------

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


# 2 ノイズデータの作成 ----------------------------------------------------

# 乱数シードの設定
tf.random.set_seed(42)
np.random.seed(42)

# ノイズ係数の設定
noise_factor = 0.50

# 訓練データ
# --- ノイズ追加
X_train_AE_noisy = X_train_AE.copy() + \
                   noise_factor * \
                   np.random.normal(loc=0.0, scale=1.0, size=X_train_AE.shape)

# テストデータ
# --- ノイズ追加
X_test_AE_noisy = X_test_AE.copy() + \
                  noise_factor * \
                  np.random.normal(loc=0.0, scale=1.0, size=X_test_AE.shape)


# 3 ノイズ除去の未完備オートエンコーダ -------------------------------------

# 元のデータセットの次元数
# --- 29
X_train.shape

# モデル構築
model = Sequential()
model.add(Dense(units=27, activation='linear', input_dim=29))
model.add(Dense(units=29, activation='linear'))

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
# --- 平均適合率は0.69（書籍より）
# --- 変動係数は0.90（書籍より）
anomalyScoresAE = anomalyScores(X_test, predictions)
preds = plotResults(y_test, anomalyScoresAE, True)

# 後処理
model.reset_states()


# 4 ノイズ除去の過完備オートエンコーダ ------------------------------

# 元のデータセットの次元数
# --- 29
X_train.shape

# モデル構築
model = Sequential()
model.add(Dense(units=40, activation='linear',
                activity_regularizer=regularizers.l1(10e-5), input_dim=29))
model.add(Dropout(0.05))
model.add(Dense(units=29, activation='linear'))

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
# --- 平均適合率は0.08（書籍より）
# --- 変動係数は0.79（書籍より）
anomalyScoresAE = anomalyScores(X_test, predictions)
preds = plotResults(y_test, anomalyScoresAE, True)

# 後処理
model.reset_states()


# 5 Reluを用いたノイズ除去の過完備オートエンコーダ ------------------------------

# 元のデータセットの次元数
# --- 29
X_train.shape

# モデル構築
model = Sequential()
model.add(Dense(units=40, activation='relu',
                activity_regularizer=regularizers.l1(10e-5), input_dim=29))
model.add(Dropout(0.05))
model.add(Dense(units=29, activation='linear'))

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
# --- 平均適合率は0.69（書籍より）
# --- 変動係数は0.90（書籍より）
anomalyScoresAE = anomalyScores(X_test, predictions)
preds = plotResults(y_test, anomalyScoresAE, True)

# 後処理
model.reset_states()
