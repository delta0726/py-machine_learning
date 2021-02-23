# ******************************************************************************
# Chapter   : 5 ロジスティック回帰
# Title     : 5-5 ロジスティック回帰による機械学習（Recipe36)
# Created by: Owner
# Created on: 2020//
# Page      : P - P
# ******************************************************************************

# ＜概要＞
# - ロジスティック回帰を用いて機械学習のフローを確認する


# ＜目次＞
# 0 準備
# 1 データ分割
# 2 機械学習の一連処理


# 0 準備 ------------------------------------------------------------------------------------------

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# パスの取得
current_path = os.getcwd()
file = os.path.sep.join(['', 'csv', 'pima-indians-diabetes.csv'])

# 列名指定
column_names = ['pregnancy_x',
                'plasma_con',
                'blood_pressure',
                'skin_mm',
                'insulin',
                'bmi',
                'pedigree_func',
                'age',
                'target']

# データ取得
all_data = pd.read_csv(current_path + file,  names=column_names)

# 特徴量の列名
feature_names = column_names[:-1]

# データ格納
X = all_data[feature_names]
y = all_data['target']


# 1 データ分割 --------------------------------------------------------------------------------------

# データ分割
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)

# 確認
X_train.shape
X_test.shape


# 2 機械学習の一連処理 -----------------------------------------------------------------------------

# インスタンス生成
lr = LogisticRegression()
vars(lr)

# 学習
lr.fit(X_train, y_train)
vars(lr)

# 予測
y_pred = lr.predict(X_test)
y_pred[:10]

# モデル評価
accuracy_score(y_true=y_test, y_pred=y_pred)
