# ******************************************************************************
# Chapter   : 3 次元削減 - PCAから性能テストまで
# Title     : 3-3 分解に因子分析を利用する（Recipe20)
# Created by: Owner
# Created on: 2020/12/25
# Page      : P87 - P88
# ******************************************************************************

# ＜概要＞
# - 因子分析も次元削減の方法の1つ
#   --- データセットの特徴量に影響を与える暗黙的な特徴量が存在することを仮定する
#   --- プログラミング的な操作はPCAと同じ
# - 因子分析は確率的な変換であるため、観測値の対数尤度を調べたり、モデル間での尤度比較ができる


# ＜目次＞
# 0 準備
# 1 因子分析の実行
# 2 主成分分析と比較
# 3 プロット作成


# 0 準備 ------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA

# データロード
iris = datasets.load_iris()

# データ格納
iris_X = iris.data
y = iris.target


# 1 因子分析の実行 --------------------------------------------------------------------------------

# インスタンスの生成
fa = FactorAnalysis(n_components=2)
vars(fa)

# 学習
iris_fa = fa.fit_transform(iris_X)

# 確認
iris_fa[:5]
vars(fa)


# 2 主成分分析と比較 --------------------------------------------------------------------------------

# インスタンスの生成
pca = PCA(n_components=2)
vars(pca)

# 学習
iris_pca = pca.fit_transform(iris_X)

# 確認
iris_pca[:5]
vars(pca)


# 3 プロット作成 ----------------------------------------------------------------------------------

# 関数定義
# --- faとpcaを比較
def plot_compare(array_fa, array_pca):

    # プロット設定
    fig = plt.figure(figsize=(20, 7))

    # プロット1
    ax1 = fig.add_subplot(121)
    ax1.scatter(array_fa[:, 0], array_fa[:, 1], c=y, s=40)
    ax1.set_title('PCA')

    # プロット2
    ax2 = fig.add_subplot(122)
    ax2.scatter(array_pca[:, 0], array_pca[:, 1], c=y, s=40)
    ax2.set_title('FA')

    # 表示
    plt.show()


# 実行
plot_compare(iris_fa, iris_pca)
