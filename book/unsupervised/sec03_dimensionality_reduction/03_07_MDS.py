# ***************************************************************************************
# Book      : Pythonではじめる教師なし学習
# Chapter   : 3.次元削減
# Title     : 3.7 MDS(多次元尺度法)
# Created by: Owner
# Created on: 2020/12/31
# Page      : P85 - P86
# ***************************************************************************************


# ＜概要＞
# - MDSは元のデータセット中のデータポイント間の類似度を学習し、その類似度を用いて低次元空間にモデル化する
#   --- MDS： Multi-Dimensional Scaling


# ＜目次＞
# 0 準備
# 1 MDS(多次元尺度構成法)


# 0 準備 --------------------------------------------------------------------------------

# Main
import gzip
import os
import pickle
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import MDS


# パス設定
current_path = os.getcwd()
file = os.path.sep.join(['', 'datasets', 'mnist_data', 'mnist.pkl.gz'])

# データロード
f = gzip.open(current_path + file, 'rb')
train_set, validation_set, test_set = pickle.load(f, encoding='latin1')
f.close()

# アイテムのセット
X_train, y_train = train_set[0], train_set[1]
X_validation, y_validation = validation_set[0], validation_set[1]
X_test, y_test = test_set[0], test_set[1]

# インデックスの設定
train_index = range(0, len(X_train))
validation_index = range(len(X_train), len(X_train) + len(X_validation))
test_index = range(len(X_train) + len(X_validation),
                   len(X_train) + len(X_validation) + len(X_test))


# データフレームの作成
X_train = pd.DataFrame(data=X_train, index=train_index)
X_validation = pd.DataFrame(data=X_validation, index=validation_index)
X_test = pd.DataFrame(data=X_test, index=test_index)

# シリーズの作成
y_train = pd.Series(data=y_train, index=train_index)
y_validation = pd.Series(data=y_validation, index=validation_index)
y_test = pd.Series(data=y_test, index=test_index)


# 関数定義
# --- PC1-PC2を抽出(XY軸)
# --- ラベルを追加(ポイントのカラー)
def scatterPlot(xDF, yDF, algoName):
    temp_df = pd.DataFrame(data=xDF.loc[:, 0:1], index=xDF.index)
    temp_df = pd.concat((temp_df, yDF), axis=1, join="inner")
    temp_df.columns = ["PC1", "PC2", "Label"]
    sns.lmplot(x="PC1", y="PC2", hue="Label",
               data=temp_df, fit_reg=False)
    ax = plt.gca()
    ax.set_title("Method: " + algoName)
    plt.show()


# 1 MDS(多次元尺度構成法) -----------------------------------------------------------------------------

# パラメータの設定
n_components = 2
n_init = 12
max_iter = 1200
metric = True
n_jobs = 4
random_state = 2018

# インスタンスの作成
mds = MDS(n_components=n_components, n_init=n_init, max_iter=max_iter,
          metric=metric, n_jobs=n_jobs, random_state=random_state)

# 学習器の生成
mds.fit(X_train.loc[0:1000, :])

# 学習器の適用
X_train_mds = mds.transform(X_train.loc[0:1000, :])

# データフレームに変換
X_train_mds = pd.DataFrame(data=X_train_mds, index=train_index[0:1001])

# プロット表示
scatterPlot(X_train_mds, y_train, "Multidimensional Scaling")



