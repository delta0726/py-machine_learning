# ******************************************************************************
# Chapter   : 11 ニューラルネットワーク
# Title     : 11-2 パーセプトロン分類器（Recipe82)
# Created by: Owner
# Created on: 2020/12/30
# Page      : P317 - P324
# ******************************************************************************

# ＜概要＞
#


# ＜目次＞
# 0 準備
# 1 前処理
# 2 モデリング
# 3 クロスバリデーション
# 4 モデル評価
# 5 チューニング


# 0 準備 ------------------------------------------------------------------------------------------

import os
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, roc_auc_score


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

# データ格納
feature_names = column_names[:-1]
X = all_data[feature_names]
y = all_data['target']

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)


# 1 前処理 ---------------------------------------------------------------------------------

# インスタンス生成
# --- スケーリング
scaler = StandardScaler()

# 変換器の作成
scaler.fit(X_train)

# データ変換
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 2 モデリング ---------------------------------------------------------------------------------

# インスタンス生成
pr = Perceptron()
vars(pr)

# 学習
pr.fit(X_train_scaled, y_train)


# 3 クロスバリデーション ---------------------------------------------------------------------.

# インスタンス生成
skf = StratifiedKFold(n_splits=3)

# クロスバリデーションスコアの取得
cross_val_score(pr, X_train_scaled, y_train, cv=skf, scoring='roc_auc').mean()


# 4 モデル評価 ------------------------------------------------------------------------------

# 予測
y_pred = pr.predict(X_test_scaled)

# Accuracy
accuracy_score(y_test, y_pred)

# ROC-AUC Score
roc_auc_score(y_test, y_pred)


# 5 チューニング ----------------------------------------------------------------------------

# パラメータリスト
param_dist = {
    'alpha': [0.1, 0.01, 0.001, 0.0001],
    'penalty': [None, 'l2', 'l1', 'elasticnet'],
    'random_state': [7],
    'class_weight': ['balanced', None],
    'eta0': [0.25, 0.5, 0.75, 1.0],
    'warm_start': [True, False],
    'n_iter': [50]
}


# インスタンス生成
gs_perceptron = GridSearchCV(pr, param_dist, scoring='roc_auc', cv=skf)
gs_perceptron.fit(X_train_scaled, y_train)

# ベストパラメータ
gs_perceptron.best_params_
gs_perceptron.best_score_
