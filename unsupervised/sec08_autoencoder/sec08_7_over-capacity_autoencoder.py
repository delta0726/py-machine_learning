# ***************************************************************************************
# Book      : Pythonではじめる教師なし学習
# Chapter   : 8.オートエンコーダのハンズオン
# Title     : 8.7-10 過完備オートエンコーダ
# Created by: Owner
# Created on: 2020/1/30
# Page      : P187 - P194
# ***************************************************************************************


# ＜概要＞
# - ｢過完備オートエンコーダ｣は入力層や出力層よりも大きなノードの隠れ層を持つ
#   --- ニューラルネットワークの容量が大きいので訓練対象の観測点を記憶できる
#   --- 訓練データにオーバーフィットする（何もしなければ）
#   --- ドリップアウトや正則化でオーバーフィットを回避する（これを学ぶのが本題）

# ＜目次＞
# 0 準備
# 1 データ準備
# 2 過完備オートエンコーダ
# 3 ドロップアウトの適用
# 4 スパース性制約の正則化の適用
# 5 ドロップアウトと正則化の適用


# 0 準備 ----------------------------------------------------------------------

import os

import pandas as pd
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


# 2 過完備オートエンコーダ ----------------------------------------------------

# ＜概要＞
# - ｢未完備オートエンコーダ｣とは隠れ層が入力層よりも大きいものを指す
#   --- 丸暗記が発生して訓練データに対してオーバーフィットする


# 元のデータセットの次元数
# --- 29
X_train.shape

# モデル構築
# --- ユニット数を40に変更
model = Sequential()
model.add(Dense(units=40, activation='linear', input_dim=29))
model.add(Dense(units=29, activation='linear'))

# コンパイル
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

# 学習
# --- 完備オートエンコーダよりlossが大きくなる
# --- 元の行列をそのまま記憶するのではなく、圧縮された新しい表現を作成している
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
# --- 平均適合率は0.02（スクリプトより1回のみ）
# --- 変動係数は0.89（書籍より）
anomalyScoresAE = anomalyScores(X_test, predictions)
preds = plotResults(y_test, anomalyScoresAE, True)

# 後処理
model.reset_states()


# 3 ドロップアウトの適用 ------------------------------------------------------

# ＜概要＞
# - 過完備オートエンコーダにドロップアウトを適用してオーバーフィットを軽減


# 元のデータセットの次元数
# --- 29
X_train.shape

# モデル構築
# --- ドロップアウトでオーバーフィット軽減
model = Sequential()
model.add(Dense(units=27, activation='linear', input_dim=29))
model.add(Dropout(0.10))
model.add(Dense(units=29, activation='linear'))

# コンパイル
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

# 学習
# --- 完備オートエンコーダよりlossが大きくなる
# --- 元の行列をそのまま記憶するのではなく、圧縮された新しい表現を作成している
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
# --- 平均適合率は0.21（書籍より）
# --- 変動係数は0.40（書籍より）
anomalyScoresAE = anomalyScores(X_test, predictions)
preds = plotResults(y_test, anomalyScoresAE, True)

# 後処理
model.reset_states()


# 4 スパース性制約の正則化の適用 ---------------------------------------------------

# ＜概要＞
# - 過完備オートエンコーダに正則化を適用してオーバーフィットを軽減


# 元のデータセットの次元数
# --- 29
X_train.shape

# モデル構築
# --- 2層から3層に変更
# --- 1層目にユニット数28を追加
model = Sequential()
model.add(Dense(units=40, activation='linear',
                activity_regularizer=regularizers.l1(10e-5), input_dim=29))
model.add(Dense(units=29, activation='linear'))

# コンパイル
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

# 学習
# --- 完備オートエンコーダよりlossが大きくなる
# --- 元の行列をそのまま記憶するのではなく、圧縮された新しい表現を作成している
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
# --- 平均適合率は0.21（書籍より）
# --- 変動係数は0.99（書籍より）
anomalyScoresAE = anomalyScores(X_test, predictions)
preds = plotResults(y_test, anomalyScoresAE, True)

# 後処理
model.reset_states()


# 5 ドロップアウトと正則化の適用 ------------------------------------------------

# ＜概要＞
# - 過完備オートエンコーダにドロップアウトと正則化を適用してオーバーフィットを軽減


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
# --- 完備オートエンコーダよりlossが大きくなる
# --- 元の行列をそのまま記憶するのではなく、圧縮された新しい表現を作成している
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
# --- 平均適合率は0.24（書籍より）
# --- 変動係数は0.62（書籍より）
anomalyScoresAE = anomalyScores(X_test, predictions)
preds = plotResults(y_test, anomalyScoresAE, True)

# 後処理
model.reset_states()
