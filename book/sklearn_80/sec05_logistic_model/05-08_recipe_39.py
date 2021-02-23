# ******************************************************************************
# Chapter   : 5 ロジスティック回帰
# Title     : 5-8 ROC分析（Recipe39)
# Created by: Owner
# Created on: 2020/12/26
# Page      : P148 - P153
# ******************************************************************************

# ＜概要＞
# - NPV(Negative Positive Value)の延長線上にはROC分析として混合行列のセルを調べる標準的指標が存在する


# ＜目次＞
# 0 準備
# 1 機械学習のフロー
# 2 予測精度の評価


# 0 準備 ------------------------------------------------------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.preprocessing import binarize


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

# クラス確率の予測
y_pred_proba = lr.predict_proba(X_test)
y_pred_proba[:10]


# 2 予測精度の評価 --------------------------------------------------------------------------------

# ROCカーブの取得
# --- fpr: FP / (TP + TN)
# --- tpr:
# --- ths:
fpr, tpr, ths = roc_curve(y_test, y_pred_proba[:, 1])

# 感度をプロット
# --- X: 閾値
# --- Y: 感度
# --- 閾値が低くなるほど感度は高まる
plt.plot(ths, tpr)
plt.show()

# 閾値が0.1の時の混合行列
y_pred_th = binarize(y_pred_proba, threshold=0.1)
confusion_matrix(y_test, y_pred_th[:, 1], labels=[1, 0])
