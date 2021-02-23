# ******************************************************************************
# Chapter   : 3 次元削減 - PCAから性能テストまで
# Title     : 3-5 次元削減にTSVDを使用する（Recipe22)
# Created by: Owner
# Created on: 2020/12/25
# Page      : P91 - P94
# ******************************************************************************

# ＜概要＞
# - 打ち切り特異値分解は行列MをU/Σ/Vに因子分解する行列分解手法
# - PCAと似ているが以下のような違いがある
#   --- TSVDは因子分解がデータ行列で実行される
#   --- PCAの因子分解は共分散行列で実行される


# ＜目次＞
# 0 準備
# 1 特異値分解の実行
# 2 主成分分析と比較
# 3 プロット作成
# 4 SciPyで特異値分解


# 0 準備 ------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.decomposition import TruncatedSVD, PCA
from scipy.linalg import svd


# データロード
iris = load_iris()

# データ格納
iris_X = iris.data
y = iris.target


# 1 特異値分解の実行 ------------------------------------------------------------------------------

# インスタンス生成
svd = TruncatedSVD()
vars(svd)

# 学習
iris_svd = svd.fit_transform(iris_X)
iris_svd[:5]
vars(svd)

# 分散の寄与率
svd.explained_variance_ratio_


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
# --- kpcaとpcaを比較
def plot_compare(array_svd, array_pca):

    # プロット設定
    fig = plt.figure(figsize=(20, 7))

    # プロット1
    ax1 = fig.add_subplot(121)
    ax1.scatter(array_svd[:, 0], array_svd[:, 1], c=y, s=40)
    ax1.set_title('PCA')

    # プロット2
    ax2 = fig.add_subplot(122)
    ax2.scatter(array_pca[:, 0], array_pca[:, 1], c=y, s=40)
    ax2.set_title('FA')

    # 表示
    plt.show()


# 実行
plot_compare(iris_svd, iris_pca)


# 4 SciPyで特異値分解 ---------------------------------------------------------------------------

# 行列作成
D = np.array([[1, 2], [1, 3], [1, 4]])
D

# 行列の分解
U, S, V = svd(D, full_matrices=False)

# 確認
print('U:', U.shape)
print('S:', S.shape)
print('V:', V.shape)

# 元の行列を再構築
np.dot(U.dot(np.diag(S)), V)
