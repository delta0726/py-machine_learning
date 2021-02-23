# ******************************************************************************
# Chapter   : 5 ロジスティック回帰
# Title     : 5-10 データセットの読み込みからプロット作成までの1つにまとめる（Recipe41)
# Created by: Owner
# Created on: 2020/12/25
# Page      : P157 - P160
# ******************************************************************************

# ＜概要＞
#


# ＜目次＞
# 0 準備


# 0 準備 ------------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score


# パスの取得
current_path = os.getcwd()
file = os.path.sep.join(['', 'csv', 'breast-cancer-wisconsin.csv'])

# 列名指定
column_names = ['radius',
                'texture',
                'perimeter',
                'area',
                'smoothness',
                'compactness',
                'concavity',
                'concave points',
                'symmetry',
                'malignant']

# 特徴量の名前
feature_names = column_names[:-1]

# データ取得
all_data = pd.read_csv(current_path + file,  names=column_names)

# データ型の確認
all_data.dtypes


# 1 データ加工 -------------------------------------------------------------------------------

# データ型の変換
all_data['malignant'] = all_data['malignant'].astype(np.int)

# 変換
# --- 1は悪性腫瘍を意味する
all_data['malignant'] = np.where(all_data['malignant'] == 4, 1, 0)
all_data['malignant'].value_counts()

# データ格納
X = all_data[[col for col in feature_names if col != 'compactness']]
y = all_data.malignant


# 2 機械学習のフロー ------------------------------------------------------------------------

# データ分割
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)

# インスタンス生成
lr = LogisticRegression()

# 学習
lr.fit(X_train, y_train)

# クラス確率の予測
y_pred_proba = lr.predict_proba(X_test)


# 3 ROCカーブのプロット ------------------------------------------------------------------------」

# ROCカーブの要素計算
fpr, tpr, ths = roc_curve(y_test, y_pred_proba[:, 1])

# AUCスコア
auc_score = auc(fpr, tpr)

# ROCカーブのプロット
plt.plot(fpr, tpr, label="AUC Score: " + str(auc_score))
plt.xlabel('fpr', fontsize='15')
plt.ylabel('tpr', fontsize='15')
plt.legend(loc='best')
plt.show()

