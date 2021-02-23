# ******************************************************************************
# Chapter   : 5 ロジスティック回帰
# Title     : 5-6 混合行列を使ってロジスティック回帰の誤分類を調べる（Recipe37)
# Created by: Owner
# Created on: 2020/12/26
# Page      : P141 - P143
# ******************************************************************************

# ＜概要＞
# - 分類問題の精度評価の根底にある混合行列を確認する


# ＜目次＞
# 0 準備
# 1 機械学習のフロー
# 2 混合行列の作成


# 0 準備 ------------------------------------------------------------------------------------------

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix



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

# データ分割
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)


# 1 機械学習のフロー -----------------------------------------------------------------------------

# インスタンス生成
lr = LogisticRegression()

# 学習
lr.fit(X_train, y_train)

# 予測
y_pred = lr.predict(X_test)


# 2 混合行列の作成 -----------------------------------------------------------------------------

# ＜混合行列＞
# - X軸が｢予測値｣、Y軸が｢真の値｣を示す


# 混合行列の作成
# y_true： テストデータセット
# y_pred： ロジスティック回帰モデルの予測値
# labels： 陽性クラスの参照
confusion_matrix(y_true=y_test, y_pred=y_pred, labels=[1, 0])
