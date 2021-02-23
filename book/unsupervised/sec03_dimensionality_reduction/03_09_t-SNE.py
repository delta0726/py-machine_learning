# ***************************************************************************************
# Book      : Pythonではじめる教師なし学習
# Chapter   : 3.次元削減
# Title     : 3.9 t-SNE
# Created by: Owner
# Created on: 2020/12/31
# Page      : P87 - P89
# ***************************************************************************************

# ＜概要＞
#


# ＜目次＞
# 0 準備
# 1 t-SNE


# 0 準備 --------------------------------------------------------------------------------

# Main
import gzip
import os
import pickle
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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


# 1 PCAの適用 -----------------------------------------------------------------------

# パラメータの設定
n_components = 784
whiten = False
random_state = 2018

# インスタンスの作成
pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
vars(pca)

# PCAの適用
X_train_PCA = pca.fit_transform(X_train)


# 2 t-SNE ---------------------------------------------------------------------------

# パラメータの設定
n_components = 2
learning_rate = 300
perplexity = 30
early_exaggeration = 12
init = 'random'
random_state = 2018

# インスタンス生成
tSNE = TSNE(n_components=n_components, learning_rate=learning_rate,
            perplexity=perplexity, early_exaggeration=early_exaggeration,
            init=init, random_state=random_state)

# 学習器の作成
tSNE.fit(X_train_PCA.loc[:5000, :9])

# 学習器の適用
X_train_tSNE = tSNE.transform(X_train_PCA.loc[:5000, :9])

# データフレームに変換
X_train_tSNE = pd.DataFrame(data=X_train_tSNE, index=train_index[:5001])

# プロット表示
scatterPlot(X_train_tSNE, y_train, "t-SNE")
