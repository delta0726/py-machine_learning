# ***************************************************************************************
# Book      : Pythonではじめる教師なし学習
# Chapter   : 3.次元削減
# Title     : 3.4 特異値分解
# Created by: Owner
# Created on: 2020/12/31
# Page      : P81 - P82
# ***************************************************************************************

# ＜概要＞
# - 特異値分解(SVD：Singular Value Decomposition)の手順は以下の通り
#   --- 1.元の特徴量行列のランクよりも小さいランクを持つ行列を作る
#   --- 2.小さなランクの行列のベクトルの一部を線形結合として元の行列が再編成できるようにする


# ＜目次＞
# 0 準備
# 1 学習器の作成
# 2 モデル評価


# 0 準備 --------------------------------------------------------------------------------

# Main
import gzip
import os
import pickle
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import TruncatedSVD


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


# 1 学習器の作成 ------------------------------------------------------------------------------

# パラメータの設定
n_components = 200
algorithm = 'randomized'
n_iter = 5
random_state = 2018

# インスタンス生成
svd = TruncatedSVD(n_components=n_components, algorithm=algorithm,
                   n_iter=n_iter, random_state=random_state)

# 確認
vars(svd)

# 学習器の作成
svd.fit(X_train)

# 学習器の適用
X_train_svd = svd.transform(X_train)


# 2 モデル評価 ------------------------------------------------------------------------------

# データフレームに変換
X_train_svd = pd.DataFrame(data=X_train_svd, index=train_index)

# プロット表示
scatterPlot(X_train_svd, y_train, "Singular Value Decomposition")
