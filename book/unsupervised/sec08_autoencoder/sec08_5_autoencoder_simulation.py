# ***************************************************************************************
# Book      : Pythonではじめる教師なし学習
# Chapter   : 8.オートエンコーダのハンズオン
# Title     : 8.5 線形活性化関数を用いた2層未完備オートエンコーダ
# Created by: Owner
# Created on: 2020/1/28
# Page      : P179 - P184
# ***************************************************************************************


# ＜概要＞
# - ｢完備オートエンコーダ｣と｢未完備オートエンコーダ｣の違いを学ぶ
# - ネットワークの構築方法によって結果が大きく変わることを知る
#   --- 最適なネットワークの構築にはトライ＆エラーが必要
#   --- 実験やチューニングを効率的に行う仕組みが必要


# ＜目次＞
# 0 準備
# 1 データ準備
# 2 未完備オートエンコーダ
# 3 ノード数を増やす
# 4 層を増やす


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


# 2 未完備オートエンコーダ ----------------------------------------------------

# ＜概要＞
# - ｢未完備オートエンコーダ｣とは隠れ層が入力層よりも小さいものを指す
#   --- エンコーダに制約をかけることで、丸暗記ではなく新しい表現を作り出す
#   ---- 隠れ層のノード数を29から20に変更


# 元のデータセットの次元数
# --- 29
X_train.shape

# モデル構築
# --- ユニット数を20に変更
model = Sequential()
model.add(Dense(units=20, activation='linear', input_dim=29))
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
# --- 平均適合率は0.29（完備オートエンコーダの場合と同程度）
# --- 変動係数は0.03（完備オートエンコーダの場合より大幅低下）
anomalyScoresAE = anomalyScores(X_test, predictions)
preds = plotResults(y_test, anomalyScoresAE, True)

# 後処理
model.reset_states()


# 3 ノード数を増やす ------------------------------------------------------

# ＜概要＞
# - 隠れ層のノード数を20から27に変更する


# 元のデータセットの次元数
# --- 29
X_train.shape

# モデル構築
# --- ユニット数を27に変更
model = Sequential()
model.add(Dense(units=27, activation='linear', input_dim=29))
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
# --- 平均適合率は0.53
# --- 変動係数は0.50（ノード数20の場合より増加）
anomalyScoresAE = anomalyScores(X_test, predictions)
preds = plotResults(y_test, anomalyScoresAE, True)

# 後処理
model.reset_states()


# 4 層を増やす ------------------------------------------------------------

# ＜概要＞
# - 隠れ層のノード数を20から27に変更する


# 元のデータセットの次元数
# --- 29
X_train.shape

# モデル構築
# --- 2層から3層に変更
# --- 1層目にユニット数28を追加
model = Sequential()
model.add(Dense(units=28, activation='linear', input_dim=29))
model.add(Dense(units=27, activation='linear'))
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
# --- 平均適合率は0.36
# --- 変動係数は0.94
anomalyScoresAE = anomalyScores(X_test, predictions)
preds = plotResults(y_test, anomalyScoresAE, True)

# 後処理
model.reset_states()
