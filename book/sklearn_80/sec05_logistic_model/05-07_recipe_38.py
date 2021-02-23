# ******************************************************************************
# Chapter   : 5 ロジスティック回帰
# Title     : 5-7 ロジスティック回帰で分類のしきい値を変化させる（Recipe38)
# Created by: Owner
# Created on: 2020/12/25
# Page      : P143 - P148
# ******************************************************************************

# ＜概要＞
# - 分類問題のしきい値を変更することによる影響を確認する
#   --- デフォルトは0.5だが、閾値は問題に応じて設定されるべきもの


# ＜目次＞
# 0 準備
# 1 機械学習のフロー
# 2 クラス確率の可視化
# 3 クラス分類のしきい値の変更
# 4 NPVプロットの作成


# 0 準備 ------------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import binarize
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
# --- クラス確率で表示
y_pred_proba = lr.predict_proba(X_test)
y_pred_proba[:10]


# 2 クラス確率の可視化 ----------------------------------------------------------------------------

# ヒストグラム作成
# --- クラス確率
pd.Series(y_pred_proba[:, 1]).hist()
plt.show()

# ヒストグラム作成
# --- ラベル
all_data['target'].hist()
plt.show()


# 3 クラス分類のしきい値の変更 --------------------------------------------------------------------

# ** threshold=0.5 *********************

# クラス確率の確認
y_pred_proba[:9]

# バイナリ変換
# --- しきい値を0.5に指定
y_pred_default = binarize(y_pred_proba, threshold=0.5)
y_pred_default[:9]

# 混合行列の作成
confusion_matrix(y_true=y_test, y_pred=y_pred_default[:, 1], labels=[1, 0])


# ** threshold=0.2 *********************

# クラス確率の確認
y_pred_proba[:9]

# バイナリ変換
# --- しきい値を0.2に指定
# --- どちらも1と判定されてしまう（要修正）
y_pred_low = binarize(y_pred_proba[:, 1], threshold=0.2)
y_pred_low[:9]

# 混合行列の作成
confusion_matrix(y_true=y_test, y_pred=y_pred_low[:, 1], labels=[1, 0])


# 4 NPVプロットの作成 -------------------------------------------------------------------------

# ＜NPVとは＞
# - 陰性的中率(NPV: Negative Predictive Value)は適合率(TP/(TP+FN))を応用したもの
# - しきい値の水準ごとに適合率を算出


# 関数定義
# --- NPV
def npv_func(threshold=0.5):
    y_pred = binarize(y_pred_proba, threshold=threshold)
    second_column = confusion_matrix(y_test, y_pred[:, 1], labels=[1, 0])[:, 1]
    npv = second_column[1] / second_column.sum()
    return npv


# 閾値のパターン
ths = np.arange(0, 1, 0.05)

# オブジェクト準備
npvs = []
for th in np.arange(0, 1, 0.05):
    npvs.append(npv_func(th))


# NPVプロット作成
plt.plot(ths, npvs)
plt.show()
