# ***************************************************************************************
# Book      : Pythonではじめる教師なし学習
# Title     : 2 機械学習プロジェクトのはじめから終わりまで
# Objective : TODO
# Created by: Owner
# Created on: 2020/11/7
# ***************************************************************************************


# ＜目次＞
# 2.3 データ準備
# 2.4 モデル準備
# 2.5 機械学習モデル(Part1)
# 2.7 機械学習モデル(Part2)


# 2.3 データ準備 - -----------------------------------------------------------------


"""Main"""
import os

import numpy as np
import pandas as pd

'''Data Viz'''
import matplotlib.pyplot as plt
import seaborn as sns

color = sns.color_palette()

'''Data Prep'''
from sklearn import preprocessing as pp
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc

'''Algos'''
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb


# データ概要 ************************************************************************

# データの読み込み
current_path = os.getcwd()
file = os.path.sep.join(['', 'book', 'unsupervised', 'datasets', 'credit_card_data', 'credit_card.csv'])
data = pd.read_csv(current_path + file)


# データ構造
data.shape
data.columns
data.info()


# 先頭データ
data.head()


# 基本統計量
data.describe()


# 不正ラベルの数
# --- Class列の1をカウント
print("Number of fraudulent transactions:", data['Class'].sum())


# NA確認
# --- 0件
# --- 欠損値処理は必要ない
nanCounter = np.isnan(data).sum()


# ユニークレコードの件数
# --- 多くの列がユニーク
# --- Classは1/0の2つのみ
distinctCounter = data.apply(lambda x: len(x.unique()))


# ** データ作成 ************************************************************************

# 特徴量行列(X)
dataX = data.copy().drop(['Class'], axis=1)


# ラベル配列(Y)
dataY = data['Class'].copy()


# データ基準化
# --- Time列を削除
# --- 基準化
featuresToScale = dataX.drop(['Time'], axis=1).columns
sX = pp.StandardScaler(copy=True)
dataX.loc[:, featuresToScale] = sX.fit_transform(dataX[featuresToScale])


# データ確認
# --- 平均：0 標準偏差：1
scalingFactors = pd.DataFrame(data=[sX.mean_, sX.scale_], index=['Mean', 'StDev'], columns=featuresToScale)
scalingFactors.describe()


# ** 特徴量エンジニアリングと特徴量選択 -------------------------------------------------


# 相関係数行列のオブジェクト準備
correlationMatrix = pd.DataFrame(data=[], index=dataX.columns, columns=dataX.columns)

# 相関係数行列の作成
for i in dataX.columns:
    for j in dataX.columns:
        correlationMatrix.loc[i, j] = np.round(pearsonr(dataX.loc[:, i], dataX.loc[:, j])[0], 2)


# ファイル保存
# --- 相関係数行列
# --- 特徴量間の相関は十分に低いので特徴量エンジニアリングは必要ない
correlation_file = os.path.sep.join(['', 'book', 'unsupervised', 'datasets', 'credit_card_data', 'correlationMatrix.csv'])
correlationMatrix.to_csv(current_path + correlation_file)


# 関数定義
# --- プロット作成
def plot_label(df):
    count_classes = pd.value_counts(df, sort=True).sort_index()
    ax = sns.barplot(x=count_classes.index, y=tuple(count_classes / len(data)))
    ax.set_title('Frequency Percentage by Class')
    ax.set_xlabel('Class')
    ax.set_ylabel('Frequency Percentage')
    plt.show()


# データの可視化
# --- ラベルデータ
# --- 1の頻度が非常に小さく偏ったデータであることが分かる
plot_label(data['Class'])


# 2.4 モデル準備 ----------------------------------------------------------------------

# Model Preparation

# データ分割
# --- 訓練データとテストデータの分割
X_train, X_test, y_train, y_test = train_test_split(dataX,
                                                    dataY, test_size=0.33,
                                                    random_state=2018, stratify=dataY)

# レコード数の確認
# --- テストデータは指定通り33%となっている
len(X_train)
len(X_test)
len(X_test) / len(dataX)


# ラベルの割合
# --- stratifyを指定したので一致
y_train.sum() / len(y_train)
y_test.sum() / len(y_test)


# インスタンス作成
# --- クロスバリデーション
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)


# 2.5 機械学習モデル(Part1) ----------------------------------------------------------------------


# 2.5.1 ロジスティック回帰

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


# 結果格納のオブジェクト
trainingScores_rf = []
cvScores = []
predictionsBasedOnKFolds = pd.DataFrame(data=[],
                                        index=y_train.index, columns=[0, 1])

# モデルの訓練
model = logReg


# クロスバリデーション
# --- X_train_fold / y_train_fold でモデルを構築
# ---
for train_index, cv_index in k_fold.split(np.zeros(len(X_train)), y_train.ravel()):

    # データ作成
    X_train_fold, X_cv_fold = X_train.iloc[train_index, :], \
                              X_train.iloc[cv_index, :]
    y_train_fold, y_cv_fold = y_train.iloc[train_index], \
                              y_train.iloc[cv_index]

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

#


# 2.7 機械学習モデル(Part2) ----------------------------------------------------------------------

# 2.7.1 ランダムフォレスト

# ハイパーパラメータの設定
n_estimators = 10
max_features = 'auto'
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
min_weight_fraction_leaf = 0.0
max_leaf_nodes = None
bootstrap = True
oob_score = False
n_jobs = -1
random_state = 2018
class_weight = 'balanced'


# インスタンス作成
RFC = RandomForestClassifier(n_estimators=n_estimators,
                             max_features=max_features, max_depth=max_depth,
                             min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                             min_weight_fraction_leaf=min_weight_fraction_leaf,
                             max_leaf_nodes=max_leaf_nodes, bootstrap=bootstrap,
                             oob_score=oob_score, n_jobs=n_jobs, random_state=random_state,
                             class_weight=class_weight)


# 結果格納用オブジェクト
trainingScores = []
cvScores = []
predictionsBasedOnKFolds = pd.DataFrame(data=[],
                                        index=y_train.index, columns=[0, 1])

model = RFC

for train_index, cv_index in k_fold.split(np.zeros(len(X_train)),
                                          y_train.ravel()):

    # データ抽出
    X_train_fold, X_cv_fold = X_train.iloc[train_index, :], \
                              X_train.iloc[cv_index, :]
    y_train_fold, y_cv_fold = y_train.iloc[train_index], \
                              y_train.iloc[cv_index]

    # 学習
    model.fit(X_train_fold, y_train_fold)

    # モデル評価
    loglossTraining = log_loss(y_train_fold,
                               model.predict_proba(X_train_fold)[:, 1])

    # 結果追加
    trainingScores_rf.append(loglossTraining)

    predictionsBasedOnKFolds.loc[X_cv_fold.index, :] = model.predict_proba(X_cv_fold)
    loglossCV = log_loss(y_cv_fold, \
                         predictionsBasedOnKFolds.loc[X_cv_fold.index, 1])
    cvScores.append(loglossCV)

    print('Training Log Loss: ', loglossTraining)
    print('CV Log Loss: ', loglossCV)


loglossRandomForestsClassifier = log_loss(y_train,
                                          predictionsBasedOnKFolds.loc[:, 1])
print('Random Forests Log Loss: ', loglossRandomForestsClassifier)

#

preds = pd.concat([y_train, predictionsBasedOnKFolds.loc[:, 1]], axis=1)
preds.columns = ['trueLabel', 'prediction']
predictionsBasedOnKFoldsRandomForests = preds.copy()

precision, recall, thresholds = precision_recall_curve(preds['trueLabel'],
                                                       preds['prediction'])
average_precision = average_precision_score(preds['trueLabel'],
                                            preds['prediction'])

plt.step(recall, precision, color='k', alpha=0.7, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])

plt.title('Precision-Recall curve: Average Precision = {0:0.2f}'.format(
    average_precision))

fpr, tpr, thresholds = roc_curve(preds['trueLabel'], preds['prediction'])
areaUnderROC = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic: \
          Area under the curve = {0:0.2f}'.format(
    areaUnderROC))
plt.legend(loc="lower right")
plt.show()

#

params_xGB = {
    'nthread': 16,  # number of cores
    'learning rate': 0.3,  # range 0 to 1, default 0.3
    'gamma': 0,  # range 0 to infinity, default 0 
    # increase to reduce complexity (increase bias, reduce variance)
    'max_depth': 6,  # range 1 to infinity, default 6
    'min_child_weight': 1,  # range 0 to infinity, default 1
    'max_delta_step': 0,  # range 0 to infinity, default 0
    'subsample': 1.0,  # range 0 to 1, default 1
    # subsample ratio of the training examples
    'colsample_bytree': 1.0,  # range 0 to 1, default 1 
    # subsample ratio of features
    'objective': 'binary:logistic',
    'num_class': 1,
    'eval_metric': 'logloss',
    'seed': 2018,
    'silent': 1
}

#

trainingScores_rf = []
cvScores = []
predictionsBasedOnKFolds = pd.DataFrame(data=[],
                                        index=y_train.index, columns=['prediction'])

for train_index, cv_index in k_fold.split(np.zeros(len(X_train)),
                                          y_train.ravel()):
    X_train_fold, X_cv_fold = X_train.iloc[train_index, :], \
                              X_train.iloc[cv_index, :]
    y_train_fold, y_cv_fold = y_train.iloc[train_index], \
                              y_train.iloc[cv_index]

    dtrain = xgb.DMatrix(data=X_train_fold, label=y_train_fold)
    dCV = xgb.DMatrix(data=X_cv_fold)

    bst = xgb.cv(params_xGB, dtrain, num_boost_round=2000,
                 nfold=5, early_stopping_rounds=200, verbose_eval=50)

    best_rounds = np.argmin(bst['test-logloss-mean'])
    bst = xgb.train(params_xGB, dtrain, best_rounds)

    loglossTraining = log_loss(y_train_fold, bst.predict(dtrain))
    trainingScores_rf.append(loglossTraining)

    predictionsBasedOnKFolds.loc[X_cv_fold.index, 'prediction'] = \
        bst.predict(dCV)
    loglossCV = log_loss(y_cv_fold, \
                         predictionsBasedOnKFolds.loc[X_cv_fold.index, 'prediction'])
    cvScores.append(loglossCV)

    print('Training Log Loss: ', loglossTraining)
    print('CV Log Loss: ', loglossCV)

loglossXGBoostGradientBoosting = \
    log_loss(y_train, predictionsBasedOnKFolds.loc[:, 'prediction'])
print('XGBoost Gradient Boosting Log Loss: ', loglossXGBoostGradientBoosting)

#

preds = pd.concat([y_train, predictionsBasedOnKFolds.loc[:, 'prediction']], axis=1)
preds.columns = ['trueLabel', 'prediction']
predictionsBasedOnKFoldsXGBoostGradientBoosting = preds.copy()

precision, recall, thresholds = \
    precision_recall_curve(preds['trueLabel'], preds['prediction'])
average_precision = \
    average_precision_score(preds['trueLabel'], preds['prediction'])

plt.step(recall, precision, color='k', alpha=0.7, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])

plt.title('Precision-Recall curve: Average Precision = {0:0.2f}'.format(
    average_precision))

fpr, tpr, thresholds = roc_curve(preds['trueLabel'], preds['prediction'])
areaUnderROC = auc(fpr, tpr)

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

#

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

#

trainingScores_rf = []
cvScores = []
predictionsBasedOnKFolds = pd.DataFrame(data=[],
                                        index=y_train.index, columns=['prediction'])

for train_index, cv_index in k_fold.split(np.zeros(len(X_train)),
                                          y_train.ravel()):
    X_train_fold, X_cv_fold = X_train.iloc[train_index, :], \
                              X_train.iloc[cv_index, :]
    y_train_fold, y_cv_fold = y_train.iloc[train_index], \
                              y_train.iloc[cv_index]

    lgb_train = lgb.Dataset(X_train_fold, y_train_fold)
    lgb_eval = lgb.Dataset(X_cv_fold, y_cv_fold, reference=lgb_train)
    gbm = lgb.train(params_lightGB, lgb_train, num_boost_round=2000,
                    valid_sets=lgb_eval, early_stopping_rounds=200)

    loglossTraining = log_loss(y_train_fold, \
                               gbm.predict(X_train_fold, num_iteration=gbm.best_iteration))
    trainingScores_rf.append(loglossTraining)

    predictionsBasedOnKFolds.loc[X_cv_fold.index, 'prediction'] = \
        gbm.predict(X_cv_fold, num_iteration=gbm.best_iteration)
    loglossCV = log_loss(y_cv_fold, \
                         predictionsBasedOnKFolds.loc[X_cv_fold.index, 'prediction'])
    cvScores.append(loglossCV)

    print('Training Log Loss: ', loglossTraining)
    print('CV Log Loss: ', loglossCV)

loglossLightGBMGradientBoosting = \
    log_loss(y_train, predictionsBasedOnKFolds.loc[:, 'prediction'])
print('LightGBM Gradient Boosting Log Loss: ', loglossLightGBMGradientBoosting)

#

preds = pd.concat([y_train, predictionsBasedOnKFolds.loc[:, 'prediction']], axis=1)
preds.columns = ['trueLabel', 'prediction']
predictionsBasedOnKFoldsLightGBMGradientBoosting = preds.copy()

precision, recall, thresholds = \
    precision_recall_curve(preds['trueLabel'], preds['prediction'])
average_precision = \
    average_precision_score(preds['trueLabel'], preds['prediction'])

plt.step(recall, precision, color='k', alpha=0.7, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])

plt.title('Precision-Recall curve: Average Precision = {0:0.2f}'.format(
    average_precision))

fpr, tpr, thresholds = roc_curve(preds['trueLabel'], preds['prediction'])
areaUnderROC = auc(fpr, tpr)

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

#

# Test Set Evaluation

#

predictionsTestSetLogisticRegression = \
    pd.DataFrame(data=[], index=y_test.index, columns=['prediction'])
predictionsTestSetLogisticRegression.loc[:, 'prediction'] = \
    logReg.predict_proba(X_test)[:, 1]
logLossTestSetLogisticRegression = \
    log_loss(y_test, predictionsTestSetLogisticRegression)

#

predictionsTestSetRandomForests = \
    pd.DataFrame(data=[], index=y_test.index, columns=['prediction'])
predictionsTestSetRandomForests.loc[:, 'prediction'] = \
    RFC.predict_proba(X_test)[:, 1]
logLossTestSetRandomForests = \
    log_loss(y_test, predictionsTestSetRandomForests)

#

predictionsTestSetXGBoostGradientBoosting = \
    pd.DataFrame(data=[], index=y_test.index, columns=['prediction'])
dtest = xgb.DMatrix(data=X_test)
predictionsTestSetXGBoostGradientBoosting.loc[:, 'prediction'] = \
    bst.predict(dtest)
logLossTestSetXGBoostGradientBoosting = \
    log_loss(y_test, predictionsTestSetXGBoostGradientBoosting)

#

predictionsTestSetLightGBMGradientBoosting = \
    pd.DataFrame(data=[], index=y_test.index, columns=['prediction'])
predictionsTestSetLightGBMGradientBoosting.loc[:, 'prediction'] = \
    gbm.predict(X_test, num_iteration=gbm.best_iteration)
logLossTestSetLightGBMGradientBoosting = \
    log_loss(y_test, predictionsTestSetLightGBMGradientBoosting)

#

print("Log Loss of Logistic Regression on Test Set: ", \
      logLossTestSetLogisticRegression)
print("Log Loss of Random Forests on Test Set: ", \
      logLossTestSetRandomForests)
print("Log Loss of XGBoost Gradient Boosting on Test Set: ", \
      logLossTestSetXGBoostGradientBoosting)
print("Log Loss of LightGBM Gradient Boosting on Test Set: ", \
      logLossTestSetLightGBMGradientBoosting)

#

precision, recall, thresholds = \
    precision_recall_curve(y_test, predictionsTestSetLogisticRegression)
average_precision = \
    average_precision_score(y_test, predictionsTestSetLogisticRegression)

plt.step(recall, precision, color='k', alpha=0.7, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])

plt.title('Precision-Recall curve: Average Precision = {0:0.2f}'.format(
    average_precision))

fpr, tpr, thresholds = roc_curve(y_test, predictionsTestSetLogisticRegression)
areaUnderROC = auc(fpr, tpr)

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

#

precision, recall, thresholds = \
    precision_recall_curve(y_test, predictionsTestSetRandomForests)
average_precision = \
    average_precision_score(y_test, predictionsTestSetRandomForests)

plt.step(recall, precision, color='k', alpha=0.7, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])

plt.title('Precision-Recall curve: Average Precision = {0:0.2f}'.format(
    average_precision))

fpr, tpr, thresholds = roc_curve(y_test, predictionsTestSetRandomForests)
areaUnderROC = auc(fpr, tpr)

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

#

precision, recall, thresholds = \
    precision_recall_curve(y_test, predictionsTestSetXGBoostGradientBoosting)
average_precision = \
    average_precision_score(y_test, predictionsTestSetXGBoostGradientBoosting)

plt.step(recall, precision, color='k', alpha=0.7, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])

plt.title('Precision-Recall curve: Average Precision = {0:0.2f}'.format(
    average_precision))

fpr, tpr, thresholds = \
    roc_curve(y_test, predictionsTestSetXGBoostGradientBoosting)
areaUnderROC = auc(fpr, tpr)

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

#

precision, recall, thresholds = \
    precision_recall_curve(y_test, predictionsTestSetLightGBMGradientBoosting)
average_precision = \
    average_precision_score(y_test, predictionsTestSetLightGBMGradientBoosting)

plt.step(recall, precision, color='k', alpha=0.7, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])

plt.title('Precision-Recall curve: Average Precision = {0:0.2f}'.format(
    average_precision))

fpr, tpr, thresholds = \
    roc_curve(y_test, predictionsTestSetLightGBMGradientBoosting)
areaUnderROC = auc(fpr, tpr)

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

#

# Ensemble

#

predictionsBasedOnKFoldsFourModels = pd.DataFrame(data=[], index=y_train.index)
predictionsBasedOnKFoldsFourModels = predictionsBasedOnKFoldsFourModels.join(
    predictionsBasedOnKFoldsLogisticRegression['prediction'].astype(float), \
    how='left').join(predictionsBasedOnKFoldsRandomForests['prediction'] \
                     .astype(float), how='left', rsuffix="2").join( \
    predictionsBasedOnKFoldsXGBoostGradientBoosting['prediction'].astype(float), \
    how='left', rsuffix="3").join( \
    predictionsBasedOnKFoldsLightGBMGradientBoosting['prediction'].astype(float), \
    how='left', rsuffix="4")
predictionsBasedOnKFoldsFourModels.columns = \
    ['predsLR', 'predsRF', 'predsXGB', 'predsLightGBM']

#

X_trainWithPredictions = \
    X_train.merge(predictionsBasedOnKFoldsFourModels,
                  left_index=True, right_index=True)

#

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

#

trainingScores_rf = []
cvScores = []
predictionsBasedOnKFoldsEnsemble = \
    pd.DataFrame(data=[], index=y_train.index, columns=['prediction'])

for train_index, cv_index in k_fold.split(np.zeros(len(X_train)), \
                                          y_train.ravel()):
    X_train_fold, X_cv_fold = \
        X_trainWithPredictions.iloc[train_index, :], \
        X_trainWithPredictions.iloc[cv_index, :]
    y_train_fold, y_cv_fold = y_train.iloc[train_index], y_train.iloc[cv_index]

    lgb_train = lgb.Dataset(X_train_fold, y_train_fold)
    lgb_eval = lgb.Dataset(X_cv_fold, y_cv_fold, reference=lgb_train)
    gbm = lgb.train(params_lightGB, lgb_train, num_boost_round=2000,
                    valid_sets=lgb_eval, early_stopping_rounds=200)

    loglossTraining = log_loss(y_train_fold, \
                               gbm.predict(X_train_fold, num_iteration=gbm.best_iteration))
    trainingScores_rf.append(loglossTraining)

    predictionsBasedOnKFoldsEnsemble.loc[X_cv_fold.index, 'prediction'] = \
        gbm.predict(X_cv_fold, num_iteration=gbm.best_iteration)
    loglossCV = log_loss(y_cv_fold, \
                         predictionsBasedOnKFoldsEnsemble.loc[X_cv_fold.index, 'prediction'])
    cvScores.append(loglossCV)

    print('Training Log Loss: ', loglossTraining)
    print('CV Log Loss: ', loglossCV)

loglossEnsemble = log_loss(y_train, \
                           predictionsBasedOnKFoldsEnsemble.loc[:, 'prediction'])
print('Ensemble Log Loss: ', loglossEnsemble)

#

print('Feature importances:', list(gbm.feature_importance()))

#

preds = pd.concat([y_train, predictionsBasedOnKFoldsEnsemble.loc[:, 'prediction']], axis=1)
preds.columns = ['trueLabel', 'prediction']

precision, recall, thresholds = \
    precision_recall_curve(preds['trueLabel'], preds['prediction'])
average_precision = \
    average_precision_score(preds['trueLabel'], preds['prediction'])

plt.step(recall, precision, color='k', alpha=0.7, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])

plt.title('Precision-Recall curve: Average Precision = {0:0.2f}'.format(
    average_precision))

fpr, tpr, thresholds = roc_curve(preds['trueLabel'], preds['prediction'])
areaUnderROC = auc(fpr, tpr)

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

#

scatterData = predictionsTestSetLightGBMGradientBoosting.join(y_test, how='left')
scatterData.columns = ['Predicted Probability', 'True Label']
ax = sns.regplot(x="True Label", y="Predicted Probability", color='k',
                 fit_reg=False, scatter_kws={'alpha': 0.1},
                 data=scatterData).set_title( \
    'Plot of Prediction Probabilities and the True Label')

#

scatterDataMelted = pd.melt(scatterData, "True Label", \
                            var_name="Predicted Probability")
ax = sns.stripplot(x="value", y="Predicted Probability", \
                   hue='True Label', jitter=0.4, \
                   data=scatterDataMelted).set_title( \
    'Plot of Prediction Probabilities and the True Label')

#

'''Pipeline for New Data'''
# first, import new data into a dataframe called 'newData'
# second, scale data
# newData.loc[:,featuresToScale] = sX.transform(newData[featuresToScale])
# third, predict using LightGBM
# gbm.predict(newData, num_iteration=gbm.best_iteration)
