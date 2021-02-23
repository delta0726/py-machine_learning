# ******************************************************************************
# Chapter   : 3 次元削減 - PCAから性能テストまで
# Title     : 3-4 非線形次元削減にカーネルPCAを利用する（Recipe21)
# Created by: Owner
# Created on: 2020/12/25
# Page      : P89 - P91
# ******************************************************************************

# ＜概要＞
# - PCAをはじめ統計学のほとんどの手法が線形を想定している
# - カーネルPCAは非線形の次元削減を行うことができる
#   --- カーネル関数の理解には｢カーネルPCAで利用できるカーネル関数によって分離可能なデータの生成｣を考える


# ＜目次＞
# 0 準備


# 0 準備 ------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt

from sklearn import datasets, decomposition


# データ準備
iris = datasets.load_iris()

# データ格納
iris_X = iris.data
y = iris.target


# 1 カーネルPCA --------------------------------------------------------------------------------------

# ＜ポイント＞
# - コサインカーネルは特徴空間で表された2つのサンプル間の角度を比較する

# ＜利用可能なカーネル＞
# - 多項式カーネル
# - RBFカーネル
# - シグモイドカーネル
# - コサインカーネル
# - 事前計算済みカーネル

# ＜引数＞
# - degree
#   --- 次数を指定（多項式カーネル/RBFカーネル/シグモイドカーネル）


# インスタンス生成
kpca = decomposition.KernelPCA(kernel='cosine', n_components=2)
vars(kpca)

# 学習
iris_kpca = kpca.fit_transform(iris_X)
iris_kpca[:5]


# 2 主成分分析と比較 --------------------------------------------------------------------------------

# インスタンスの生成
pca = decomposition.PCA(n_components=2)
vars(pca)

# 学習
iris_pca = pca.fit_transform(iris_X)

# 確認
iris_pca[:5]
vars(pca)


# 3 プロット作成 ----------------------------------------------------------------------------------

# 関数定義
# --- kpcaとpcaを比較
def plot_compare(array_kpca, array_pca):

    # プロット設定
    fig = plt.figure(figsize=(20, 7))

    # プロット1
    ax1 = fig.add_subplot(121)
    ax1.scatter(array_kpca[:, 0], array_kpca[:, 1], c=y, s=40)
    ax1.set_title('kernel PCA')

    # プロット2
    ax2 = fig.add_subplot(122)
    ax2.scatter(array_pca[:, 0], array_pca[:, 1], c=y, s=40)
    ax2.set_title('PCA')

    # 表示
    plt.show()


# 実行
plot_compare(iris_kpca, iris_pca)
