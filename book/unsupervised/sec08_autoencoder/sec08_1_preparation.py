# ***************************************************************************************
# Book      : Pythonではじめる教師なし学習
# Chapter   : 8.オートエンコーダのハンズオン
# Title     : 8.1 データ準備
# Created by: Owner
# Created on: 2021/1/23
# Page      : P171 - P172
# ***************************************************************************************


# ＜目次＞
# 0 準備
# 1 データ準備
# 2 データ加工
# 3 分析用データの作成


# 0 準備 ----------------------------------------------------------------------

import os

import pandas as pd

from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split


# 1 データ準備 ------------------------------------------------------------------

# データ取得
# --- クレジットカードデータ
current_path = os.getcwd()
file = os.path.sep.join(['', 'datasets', 'credit_card_data', 'credit_card.csv'])
data = pd.read_csv(current_path + file)

# データ確認
data.info()

# データ格納
dataX = data.copy().drop(['Class', 'Time'], axis=1)
dataY = data['Class'].copy()


# 2 データ加工 ---------------------------------------------------------------------

# 列名取得
featuresToScale = dataX.columns

# インスタンス生成
# --- Zスコア変換
sX = pp.StandardScaler(copy=True, with_mean=True, with_std=True)

# スケーリング
dataX.loc[:, featuresToScale] = sX.fit_transform(dataX[featuresToScale])

# 確認
dataX


# 3 分析用データの作成 ---------------------------------------------------------------------

# データ分割
X_train, X_test, y_train, y_test = \
    train_test_split(dataX, dataY, test_size=0.33,
                     random_state=2018, stratify=dataY)

# 確認
# --- (190820, 29)
X_train.shape

# 訓練データを90％削除
# --- サンプリングで削除するインデックスを取得
# --- 部分的にラベル付けされたデータセットを再現
toDrop = y_train[y_train == 1].sample(frac=0.90, random_state=2018)

# 確認
# --- (297,)
toDrop.shape

# 訓練データの削除
X_train.drop(labels=toDrop.index, inplace=True)

# 確認
# --- (190523, 29)
X_train.shape
