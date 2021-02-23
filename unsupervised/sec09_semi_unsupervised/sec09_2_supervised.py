# ***************************************************************************************
# Book      : Pythonではじめる教師なし学習
# Chapter   : 9.半教師あり学習
# Title     : 9.2 教師あり学習
# Created by: Owner
# Created on: 2021/1/27
# Page      : P206 - P208
# ***************************************************************************************


# ＜目次＞
# 0 準備
# 1 データ準備
# 2 学習の準備
# 3 学習
# 4 モデル評価(訓練データ)
# 5 モデル評価(テストデータ)


# 0 準備 ----------------------------------------------------------------------

# メイン
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import preprocessing as pp
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# 共通関数
from sec09_semi_unsupervised.common_function import plotResults
from sec09_semi_unsupervised.common_function import precisionAnalysis


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

# 訓練データを90％削除
# --- サンプリングで削除するインデックスを取得
# --- 部分的にラベル付けされたデータセットを再現
toDrop = y_train[y_train == 1].sample(frac=0.90, random_state=2018)
X_train.drop(labels=toDrop.index, inplace=True)
y_train.drop(labels=toDrop.index, inplace=True)

# Foldの定義
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)


# 2 学習の準備 ------------------------------------------------------

# 変数定義
trainingScores = []
cvScores = []

# オブジェクト作成
# --- 予測値の格納用
predictionsBasedOnKFolds = pd.DataFrame(data=[], index=y_train.index,
                                        columns=['prediction'])

# パラメータ設定
# --- Light GBM
params_lightGB = {
    'task': 'train',
    'application': 'binary',
    'num_class': 1,
    'boosting': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'metric_freq': 50,
    'is_training_metric': False,
    'max_depth': 4,
    'num_leaves': 31,
    'learning_rate': 0.01,
    'feature_fraction': 1.0,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'bagging_seed': 2018,
    'verbose': 0,
    'num_threads': 16
}


# 3 学習 ------------------------------------------------------

train_index = 0
cv_index = 0
for train_index, cv_index in k_fold.split(np.zeros(len(X_train)), y_train.ravel()):

    X_train_fold, X_cv_fold = X_train.iloc[train_index, :], X_train.iloc[cv_index, :]
    y_train_fold, y_cv_fold = y_train.iloc[train_index], y_train.iloc[cv_index]

    # データ設定
    lgb_train = lgb.Dataset(X_train_fold, y_train_fold)
    lgb_eval = lgb.Dataset(X_cv_fold, y_cv_fold, reference=lgb_train)

    # 学習
    gbm = lgb.train(params_lightGB, lgb_train, num_boost_round=2000,
                    valid_sets=lgb_eval, early_stopping_rounds=200)

    loglossTraining = log_loss(y_train_fold, gbm.predict(X_train_fold,
                                                         num_iteration=gbm.best_iteration))
    trainingScores.append(loglossTraining)

    predictionsBasedOnKFolds.loc[X_cv_fold.index, 'prediction'] = \
        gbm.predict(X_cv_fold, num_iteration=gbm.best_iteration)
    loglossCV = log_loss(y_cv_fold,
                         predictionsBasedOnKFolds.loc[X_cv_fold.index, 'prediction'])
    cvScores.append(loglossCV)

    print('Training Log Loss: ', loglossTraining)
    print('CV Log Loss: ', loglossCV)


# 4 モデル評価(訓練データ) ---------------------------------------------------

# Logloss
# --- 訓練データ
loglossLightGBMGradientBoosting = log_loss(y_train,
                                           predictionsBasedOnKFolds.loc[:, 'prediction'])

# 確認
print('LightGBM Gradient Boosting Log Loss: ',
      loglossLightGBMGradientBoosting)


# 評価プロット
# --- テストデータ
preds, average_precision = plotResults(y_train,
                                       predictionsBasedOnKFolds.loc[:,'prediction'], True)


# 5 モデル評価(テストデータ) ---------------------------------------------------

# 予測データの作成
# --- テストデータ
predictions = pd.Series(data=gbm.predict(X_test, num_iteration=gbm.best_iteration),
                        index=X_test.index)

# 評価プロット
# --- テストデータ
preds, average_precision = plotResults(y_test, predictions, True)

# モデル精度の検証
preds, precision = precisionAnalysis(preds, "anomalyScore", 0.75)

# 確認
print("Precision at 75% recall", precision)
