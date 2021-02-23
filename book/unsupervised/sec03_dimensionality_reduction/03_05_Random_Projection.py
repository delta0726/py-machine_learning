# ***************************************************************************************
# Book      : Pythonではじめる教師なし学習
# Chapter   : 3.次元削減
# Title     : 3.5 ランダム射影
# Created by: Owner
# Created on: 2020/12/31
# Page      : P82 - P84
# ***************************************************************************************


# ＜概要＞
# - Johnson-Lindenstraussの補題に基づく手法
#   --- 高次元空間をはるかに低次元な空間に埋め込んだ場合に、点間の距離がほぼ保存されるというもの
#   --- 高次元空間から低次元空間に移しても元の特徴量の構造は保存されることを意味する
# - ランダム射影には以下の2つの方法がある
#   --- ガウス型ランダム射影
#   --- スパースランダム射影


# ＜目次＞
# 0 準備
# 1 ガウス型ランダム射影
# 2 スパースランダム射影


# 0 準備 --------------------------------------------------------------------------------

# Main
import gzip
import os
import pickle
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection


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


# 1 ガウス型ランダム射影 -------------------------------------------------------------------

# パラメータの設定
n_components = 'auto'
eps = 0.5
random_state = 2018

# インスタンスの作成
GRP = GaussianRandomProjection(n_components=n_components, eps=eps,
                               random_state=random_state)

# 確認
vars(GRP)

# 学習器の作成
GRP.fit(X_train)

# 学習器の適用
X_train_GRP = GRP.transform(X_train)

# データフレームに変換
X_train_GRP = pd.DataFrame(data=X_train_GRP, index=train_index)

# プロット表示
scatterPlot(X_train_GRP, y_train, "Gaussian Random Projection")


# 2 スパースランダム射影 -------------------------------------------------------------------

# パラメータの設定
n_components = 'auto'
density = 'auto'
eps = 0.5
dense_output = False
random_state = 2018

# インスタンスの作成
SRP = SparseRandomProjection(n_components=n_components,
                             density=density, eps=eps, dense_output=dense_output,
                             random_state=random_state)

# 確認
vars(SRP)

# 学習器の作成
SRP.fit(X_train)

# 学習器の適用
X_train_SRP = SRP.transform(X_train)

# データフレームに変換
X_train_SRP = pd.DataFrame(data=X_train_SRP, index=train_index)

# プロット表示
scatterPlot(X_train_SRP, y_train, "Sparse Random Projection")
