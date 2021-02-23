# ***************************************************************************************
# Book      : Pythonではじめる教師なし学習
# Chapter   : 8.オートエンコーダのハンズオン
# Title     : 8.4 最初のオートエンコーダー
# Created by: Owner
# Created on: 2020/1/27
# Page      : P173 - P179
# ***************************************************************************************


# ＜目次＞
# 0 準備
# 1 データ準備
# 2 モデル構築
# 3 学習
# 4 予測＆モデル評価
# 5 10回の試行


# 0 準備 ----------------------------------------------------------------------

import os
import time

import numpy as np
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


# 2 モデル構築 ----------------------------------------------------

# ＜概要＞
# - 2階層のオートエンコーダー
#   --- 隠れ層と出力層のみを数えて2階層（入力層は数えない）

# 元のデータセットの次元数
# --- 29
X_train.shape

# モデル構築
# --- Sequentialモデルを用いる（層を積み重ねたモデル）
# --- 1層目がエンコーダー、2層目がデコーダー
# --- それぞれ元の29次元を維持（完備オートエンコーダー）
# --- それぞれ線形活性化関数を用いる
model = Sequential()
model.add(Dense(units=29, activation='linear', input_dim=29))
model.add(Dense(units=29, activation='linear'))

# コンパイル
# --- 元の行列と再編成後の行列の再編成誤差に基づいてモデルを評価（ここでは平均二乗誤差を用いる）
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])


# 3 学習 -----------------------------------------------------------------------

# 変数定義
# --- エポック数： データセットを使って学習する階数
# --- バッチサイズ： ニューラルネットワークが1度勾配を更新する際に学習するサンプル数
num_epochs = 10
batch_size = 32

# 学習
# --- xとyのそれぞれに元の特徴量行列を入力している
# --- 完備オートエンコーダなのでロスが非常に小さくAccuracyも高い
history = model.fit(x=X_train_AE, y=X_train_AE,
                    epochs=num_epochs,
                    shuffle=True,
                    validation_data=(X_train_AE, X_train_AE),
                    verbose=1)


# 4 予測＆モデル評価 ----------------------------------------------------------------

# 予測
# --- 検証データ
predictions = model.predict(X_test, verbose=1)
predictions

# アノマリースコア
anomalyScoresAE = anomalyScores(X_test, predictions)
anomalyScoresAE

# プロット作成
preds = plotResults(y_test, anomalyScoresAE, True)
model.reset_states()


# 5 10回の試行 ----------------------------------------------------------------

# オブジェクト作成
# --- 結果格納用
test_scores = []

# ループ処理
for i in range(0, 10):

    # Progress
    print("Start Loop: ", i)
    start = time.time()

    # モデル構築
    model = Sequential()
    model.add(Dense(units=29, activation='linear',input_dim=29))
    model.add(Dense(units=29, activation='linear'))

    # コンパイル
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    # パラメータ設定
    num_epochs = 10
    batch_size = 32

    # 学習
    history = model.fit(x=X_train_AE, y=X_train_AE,
                        epochs=num_epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(X_train_AE, X_train_AE),
                        verbose=0)

    # モデル評価
    predictions = model.predict(X_test, verbose=0)
    anomalyScoresAE = anomalyScores(X_test, predictions)
    preds, avgPrecision = plotResults(y_test, anomalyScoresAE, True)

    # 結果格納
    test_scores.append(avgPrecision)

    # Progress
    elapsed_time = round(time.time() - start)
    print("Finish Loop: {0}".format(elapsed_time) + "sec")

    # 終了処理
    model.reset_states()


# 結果出力
pd.Series(test_scores)
print("平均適合率: ", np.mean(test_scores))
print("変動係数: ", np.std(test_scores) / np.mean(test_scores))
