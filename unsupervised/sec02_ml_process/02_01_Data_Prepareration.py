# ***************************************************************************************
# Book      : Pythonではじめる教師なし学習
# Chapter   : 2.機械学習プロジェクトのはじめから終わりまで
# Title     : データ準備と前処理
# Created by: Owner
# Created on: 2021/1/1
# Page      : P30 - P40
# ***************************************************************************************


# ＜概要＞
# - データ準備と機械学習を行う前の前処理を行う


# ＜目次＞
# 0 準備
# 1 データ構造の確認
# 2 特徴量行列とラベル配列の作成
# 3 相関係数行列の作成
# 4 データ可視化
# 2.5 モデリング準備


# 0 準備 --------------------------------------------------------------------------

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn import preprocessing as pp
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# データの読み込み
current_path = os.getcwd()
file = os.path.sep.join(['', 'datasets', 'credit_card_data', 'credit_card.csv'])
data = pd.read_csv(current_path + file)


# 1 データ構造の確認 ----------------------------------------------------------------

# データ構造
data.info()
data.shape
data.columns

# 先頭データ
data.head()

# 基本統計量
data.describe()

# 不正ラベルの数
# --- Class列の1をカウント
data['Class'].sum()
data['Class'].value_counts()

# NA確認
# --- データセット全体
# --- 0件なので欠損値処理は必要ない
nanCounter = np.isnan(data).sum()

# NA比率
# --- 列ごと
data.count() / len(data) * 100

# ユニークレコードの件数
# --- 多くの列がユニーク
# --- Classは1/0の2つのみ
distinctCounter = data.apply(lambda x: len(x.unique()))


# 2 特徴量行列とラベル配列の作成 ------------------------------------------------------

# 特徴量行列(X)
dataX = data.copy().drop(['Class'], axis=1)

# ラベル配列(Y)
dataY = data['Class'].copy()

# 列名取得
# --- Time列を削除
featuresToScale = dataX.drop(['Time'], axis=1).columns

# データ基準化
# --- インスタンス生成
# --- 基準化
sX = pp.StandardScaler(copy=True)
dataX.loc[:, featuresToScale] = sX.fit_transform(dataX[featuresToScale])

# データ確認
dataX.describe().T.loc[:, ['mean', 'std']]

# データ確認
# --- 各列の平均値と標準偏差
scalingFactors = pd.DataFrame(data=[sX.mean_, sX.scale_], index=['Mean', 'StDev'], columns=featuresToScale)
scalingFactors.describe()


# 3 相関係数行列の作成 -------------------------------------------------------------------

# オブジェクト準備
correlationMatrix = pd.DataFrame(data=[], index=dataX.columns, columns=dataX.columns)

# 相関係数行列の作成
# --- 特徴量間の相関は十分に低いので特徴量エンジニアリングは必要ない
for i in dataX.columns:
    for j in dataX.columns:
        correlationMatrix.loc[i, j] = np.round(pearsonr(dataX.loc[:, i], dataX.loc[:, j])[0], 2)

# ファイル保存
# --- 相関係数行列
correlation_file = os.path.sep.join(['', 'datasets', 'credit_card_data', 'correlationMatrix.csv'])
correlationMatrix.to_csv(current_path + correlation_file)


# 4 データ可視化 ------------------------------------------------------------------------

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


# 2.5 モデリング準備 ---------------------------------------------------------------

# データ分割
# --- 訓練データとテストデータの分割
# --- ラベル列で階層サンプリング
X_train, X_test, y_train, y_test = \
    train_test_split(dataX, dataY, test_size=0.33, random_state=2018, stratify=dataY)

# レコード数の確認
# --- テストデータは指定通り33%となっている
len(X_train)
len(X_test)
len(X_test) / len(dataX)

# ラベルの割合
# --- stratifyを指定したので一致
y_train.sum() / len(y_train)
y_test.sum() / len(y_test)

# クロスバリデーション
# --- インスタンス作成のみ
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
