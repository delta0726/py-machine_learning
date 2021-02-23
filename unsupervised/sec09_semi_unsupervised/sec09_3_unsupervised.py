# ***************************************************************************************
# Book      : Pythonではじめる教師なし学習
# Chapter   : 9.半教師あり学習
# Title     : 9.3 教師なしモデル
# Created by: Owner
# Created on: 2021/1/27
# Page      : P208 - P210
# ***************************************************************************************


# ＜目次＞
# 0 準備
# 1 データ準備
# 2 オーバーサンプリング
# 3 モデリング
# 4 モデル評価


# 0 準備 ----------------------------------------------------------------------

# メイン
import os

import pandas as pd
from keras import regularizers
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split

# 共通関数
from sec09_semi_unsupervised.common_function import anomalyScores
from sec09_semi_unsupervised.common_function import plotResults
from sec09_semi_unsupervised.common_function import precisionAnalysis


# 1 データ準備 ---------------------------------------------------------

# データ取得
# --- クレジットカードデータ
current_path = os.getcwd()
file = os.path.sep.join(['', 'datasets', 'credit_card_data', 'credit_card.csv'])
data = pd.read_csv(current_path + file)

# データ格納
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

# 訓練データを90％削除
# --- サンプリングで削除するインデックスを取得
# --- 部分的にラベル付けされたデータセットを再現
toDrop = y_train[y_train == 1].sample(frac=0.90, random_state=2018)
X_train.drop(labels=toDrop.index, inplace=True)
y_train.drop(labels=toDrop.index, inplace=True)

# オリジナルデータの確保
X_train_original = X_train.copy()
y_train_original = y_train.copy()
X_test_original = X_test.copy()
y_test_original = y_test.copy()


# 2 オーバーサンプリング ------------------------------------------------------

# パラメータ設定
oversample_multiplier = 100

# データ準備
X_train_oversampled = X_train.copy()
y_train_oversampled = y_train.copy()

# オーバーサンプリング
# --- X
X_train_oversampled = X_train_oversampled.append(
    [X_train_oversampled[y_train == 1]] * oversample_multiplier, ignore_index=False)

# オーバーサンプリング
# --- y
y_train_oversampled = y_train_oversampled.append(
    [y_train_oversampled[y_train == 1]] * oversample_multiplier, ignore_index=False)

# 確認
X_train_original.shape
X_train_oversampled.shape

# 元データの更新
X_train = X_train_oversampled.copy()
y_train = y_train_oversampled.copy()


# 3 モデリング -------------------------------------------------------------

# モデル定義
model = Sequential()
model.add(Dense(units=40, activation='linear',
                activity_regularizer=regularizers.l1(10e-5),
                input_dim=29, name='hidden_layer'))
model.add(Dropout(0.02))
model.add(Dense(units=29, activation='linear'))

# コンパイル
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

# パラメータ設定
num_epochs = 5
batch_size = 32

# 学習
history = model.fit(x=X_train, y=X_train,
                    epochs=num_epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_split=0.20,
                    verbose=1)

# 予測データ作成
# --- 訓練データ
# --- テストデータ
predictionsTrain = model.predict(X_train_original, verbose=1)
predictions = model.predict(X_test, verbose=1)


# 4 モデル評価 -------------------------------------------------------------

# アノマリースコア
anomalyScoresAETrain = anomalyScores(X_train_original, predictionsTrain)
anomalyScoresAE = anomalyScores(X_test, predictions)

# プロット
preds, average_precision = plotResults(y_train_original, anomalyScoresAETrain, True)
preds, average_precision = plotResults(y_test, anomalyScoresAE, True)


preds, precision = precisionAnalysis(preds, "anomalyScore", 0.75)
print("Precision at 75% recall", precision)
