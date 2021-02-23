# ***************************************************************************************
# Book      : Pythonではじめる教師なし学習
# Chapter   : 2.機械学習プロジェクトのはじめから終わりまで
# Title     : 機械学習モデル Part2
# Created by: Owner
# Created on: 2021/1/1
# Page      : P48 - P64
# ***************************************************************************************


# ＜概要＞
# - ロジスティック回帰を通してモデリングとモデル評価の流れを確認する


# ＜目次＞
# 0 準備
# 1 データ構造の確認


# 0 準備 --------------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression


# データの読み込み
current_path = os.getcwd()
file = os.path.sep.join(['', 'datasets', 'credit_card_data', 'credit_card.csv'])
data = pd.read_csv(current_path + file)

# ラベル配列(Y)
dataY = data['Class'].copy()

# 特徴量行列(X)
dataX = data.copy().drop(['Class'], axis=1)

# 列名取得
# --- Time列を削除
featuresToScale = dataX.drop(['Time'], axis=1).columns

# 特徴量行列(X)
# --- データ基準化
sX = pp.StandardScaler(copy=True)
dataX.loc[:, featuresToScale] = sX.fit_transform(dataX[featuresToScale])

# データ分割
X_train, X_test, y_train, y_test = \
    train_test_split(dataX, dataY, test_size=0.33, random_state=2018, stratify=dataY)


# クロスバリデーション
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)


# 1 ロジスティック回帰 ----------------------------------------------------------------

# ＜ポイント＞
# - penalty引数ではL2正則化(リッジ回帰)を用いる
#   --- 重要でない特徴量に0に近いウエイトを与える（実質的に特徴量選択）
# - C引数は正則化の強度を調整する
#   --- 強度が強いほどモデルの複雑性に対するペナルティが強くなる
# - ラベルが不均衡なのでclass_weight引数を指定する


# パラメータ設定
penalty = 'l2'
C = 1.0
class_weight = 'balanced'
random_state = 2018
solver = 'liblinear'
n_jobs = 1

# インスタンス作成
logReg = LogisticRegression(penalty=penalty, C=C,
                            class_weight=class_weight, random_state=random_state,
                            solver=solver, n_jobs=n_jobs)

# オブジェクト参照
model = logReg

# 結果格納のオブジェクト
trainingScores_rf = []
cvScores = []
predictionsBasedOnKFolds = pd.DataFrame(data=[],
                                        index=y_train.index, columns=[0, 1])

# クロスバリデーション
# --- X_train_fold / y_train_fold でモデルを構築
# ---
for train_index, cv_index in k_fold.split(np.zeros(len(X_train)), y_train.ravel()):

    # データ作成
    X_train_fold, X_cv_fold = X_train.iloc[train_index, :], X_train.iloc[cv_index, :]
    y_train_fold, y_cv_fold = y_train.iloc[train_index], y_train.iloc[cv_index]

    # 学習
    model.fit(X_train_fold, y_train_fold)

    # モデル評価
    # --- 訓練データ
    loglossTraining = log_loss(y_train_fold, model.predict_proba(X_train_fold)[:, 1])
    trainingScores_rf.append(loglossTraining)

    #
    predictionsBasedOnKFolds.loc[X_cv_fold.index, :] = model.predict_proba(X_cv_fold)

    # モデル評価
    # --- 評価データ
    loglossCV = log_loss(y_cv_fold, predictionsBasedOnKFolds.loc[X_cv_fold.index, 1])
    cvScores.append(loglossCV)

    # 結果表示
    print('Training Log Loss: ', loglossTraining)
    print('CV Log Loss: ', loglossCV)


# 2 モデル評価 --------------------------------------------------------------------------

# モデル評価
# --- 対数損失の算出
# --- 正解ラベルと予測値を与える
loglossLogisticRegression = log_loss(y_train, predictionsBasedOnKFolds.loc[:, 1])
print('Logistic Regression Log Loss: ', loglossLogisticRegression)


#
preds = pd.concat([y_train, predictionsBasedOnKFolds.loc[:, 1]], axis=1)
preds.columns = ['trueLabel', 'prediction']
predictionsBasedOnKFoldsLogisticRegression = preds.copy()


# メトリックの算出
# --- precision： 適合率
# --- recall   :  再現率
# ---
precision, recall, thresholds = precision_recall_curve(preds['trueLabel'],
                                                       preds['prediction'])

# 平均適合率
average_precision = average_precision_score(preds['trueLabel'],
                                            preds['prediction'])


# 3 結果のプロット表示 -------------------------------------------------------------------


# プロット表示
plt.step(recall, precision, color='k', alpha=0.7, where='post')\
    .fill_between(recall, precision, step='post', alpha=0.3, color='k')\
    .xlabel('Recall')\
    .ylabel('Precision')\
    .ylim([0.0, 1.05])\
    .xlim([0.0, 1.0])\
    .title('Precision-Recall curve: Average Precision = {0:0.2f}'.format(average_precision))\
    .show()


#

fpr, tpr, thresholds = roc_curve(preds['trueLabel'], preds['prediction'])
areaUnderROC = auc(fpr, tpr)

#

plt.figure()
plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic: \
          Area under the curve = {0:0.2f}'.format(areaUnderROC))
plt.legend(loc="lower right")
plt.show()
