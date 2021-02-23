# ******************************************************************************
# Chapter   : 5 ロジスティック回帰
# Title     : 5-9 コンテキストなしでROC曲線をプロット（Recipe40)
# Created by: Owner
# Created on: 2020/12/26
# Page      : P153 - P156
# ******************************************************************************

# ＜概要＞
# - ROCカーブは背景が分からない状態で分類器の診断をするための評価ツール
#   --- 偽陽性(FP)と偽陰性(FN)のどちらの誤検出が望ましくないかが分からないケース


# ＜目次＞
# 0 準備
# 1 機械学習のフロー
# 2 ROCカーブとAUC


# 0 準備 ------------------------------------------------------------------------------------------


import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc


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


# 2 ROCカーブとAUC --------------------------------------------------------------------------------

# ROCカーブの取得
# --- fpr: FP / (TP + TN)
# --- tpr:
# --- ths:
fpr, tpr, ths = roc_curve(y_test, y_pred_proba[:, 1])

# プロット作成
# --- ROCカーブ
plt.plot(fpr, tpr)
plt.show()

# ACUの計算
# --- AUC: Area Under the ROC Curve（ROCカーブの下の面積）
# --- ROCカーブを視覚的でなく定量的に把握する
auc(fpr, tpr)
