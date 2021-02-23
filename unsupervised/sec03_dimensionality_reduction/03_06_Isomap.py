# ***************************************************************************************
# Book      : Pythonではじめる教師なし学習
# Chapter   : 3.次元削減
# Title     : 3.6 Isomap
# Created by: Owner
# Created on: 2020/12/31
# Page      : P84 - P85
# ***************************************************************************************


# ＜概要＞
# - Isomapは非線形に次元削減を行う手法(多様体学習)の最も基本的な手法
#   --- Isomap： Isometric Mapping
# - 全ての距離は曲線距離(curved distance)もしくは測地線距離(geodesic distance)で計算する
#   --- ユーグリッド距離のような直線距離ではない
#   --- 元の特徴量集合の低次元空間への新たな埋め込みを学習する
#   --- 元データに内在する幾何構造を学習する


# ＜目次＞
# 0 準備
# 1 Isomap


# 0 準備 --------------------------------------------------------------------------------

# Main
import gzip
import os
import pickle
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import Isomap


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


# 1 Isomap ----------------------------------------------------------------------------

# パラメータの設定
n_neighbors = 5
n_components = 10
n_jobs = 4

# インスタンスの作成
isomap = Isomap(n_neighbors=n_neighbors,
                n_components=n_components, n_jobs=n_jobs)

# 学習器の生成
isomap.fit(X_train.loc[0:5000, :])

# 学習器の適用
X_train_isomap = isomap.transform(X_train)

# データフレームに変換
X_train_isomap = pd.DataFrame(data=X_train_isomap, index=train_index)

# プロット表示
scatterPlot(X_train_isomap, y_train, "Isomap")
