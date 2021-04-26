# ******************************************************************************
# Chapter   : 3 次元削減 - PCAから性能テストまで
# Title     : 3-2 PCAによる次元削減（Recipe19)
# Created by: Owner
# Created on: 2020/12/25
# Page      : P82 - P87
# ******************************************************************************


# ＜概要＞
# - PCAは統計学と線形代数を組み合わせることで次元削減に役立つ前処理ステップを生成する
#   --- 元のデータ行列を表す直交する向きの集まりを見つけ出す（新しい空間に写像）
#   --- PCAはデータの分散共分散行列を列ベクトルに変換している


# ＜目次＞
# 0 準備
# 1 PCAの実行
# 2 PCAで次元削減
# 3 PCAの可視化
# 4 分散比率から要素数を指定
# 5 PCA実行前のスケーリング


# 0 準備 ------------------------------------------------------------------------

import sklearn
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ライブラリ構成
dir(sklearn.decomposition)
dir(sklearn.pipeline)
dir(sklearn.preprocessing)


# データ準備
iris = datasets.load_iris()

# データ格納
iris_X = iris.data
iris_y = iris.target
y = iris.target


# 1 PCAの実行 -------------------------------------------------------------------

# インスタンス生成
# ---PCAの引数は他のオブジェクトに比べて少なめ
pca = decomposition.PCA()

# インスタンスの確認
print(pca)
vars(pca)

# 学習
iris_pca = pca.fit_transform(iris_X)

# 結果確認
iris_pca[:5]
vars(pca)

# 分散の説明比率
# --- PC1が約92.5％の分散を説明している
pca.explained_variance_ratio_


# 2 PCAで次元削減 ----------------------------------------------------------------

# インスタンスの生成
# --- PCの数を指定
pca = decomposition.PCA(n_components=2)
vars(pca)

# 学習
iris_X_prime = pca.fit_transform(iris_X)
iris_X_prime.shape
vars(pca)

# 分散合計
pca.explained_variance_ratio_.sum()


# 3 PCAの可視化 ------------------------------------------------------------------

def plot_pca(before, after):

    # プロット設定
    fig = plt.figure(figsize=(20, 7))

    # プロット1
    ax1 = fig.add_subplot(121)
    ax1.scatter(before[:, 0], before[:, 1], c=y, s=40)
    ax1.set_title('Before PCA')

    # プロット2
    ax2 = fig.add_subplot(122)
    ax2.scatter(after[:, 0], after[:, 1], c=y, s=40)
    ax2.set_title('After PCA')

    # 表示
    plt.show()


# プロット作成
plot_pca(iris_X, iris_X_prime)


# 4 分散比率から要素数を指定 -------------------------------------------------------

# インスタンスの生成
# --- 98%の分散を説明できるように要素数を設定
pca = decomposition.PCA(n_components=0.98)
vars(pca)

# 学習
# --- fit()とtransform()を分離
iris_X_prime = pca.fit_transform(iris_X)
vars(pca)

# 分散合計
pca.explained_variance_ratio_.sum()


# 5 PCA実行前のスケーリング -------------------------------------------------------

# ＜ポイント＞
# - PCAは事前にデータセットをスケーリングしておかなければならない
#   --- preprocessing.scale()でスケーリング


# 元データ
iris_X[:5]

# スケーリング
iris_X_scaled = preprocessing.scale(iris_X)
iris_X_scaled[:5]

# インスタンスの生成
pca = decomposition.PCA(n_components=2)
print(pca)
vars(pca)

# 学習
iris_X_scaled = pca.fit_transform(iris_X_scaled)
vars(pca)

# プロット作成
plot_pca(iris_X_prime, iris_X_scaled)


# 6 パイプラインによるスケーリング -------------------------------------------------

# ＜ポイント＞
# - パイプラインでスケーリングとPCAをつないでおくとよい


# パイプラインの構築
# --- 基準化
# --- PCA
pipe = Pipeline([('scaler', StandardScaler()),
                 ('pca', decomposition.PCA(n_components=2))])

# 確認
vars(pipe)

# 実行
iris_X_scaled = pipe.fit_transform(iris_X)
iris_X_scaled[:5]

# プロット作成
plot_pca(iris_X_prime, iris_X_scaled)
