# ***************************************************************************************
# Book      : Pythonではじめる教師なし学習
# Title     : 3 次元削減
# Objective : TODO
# Created by: Owner
# Created on: 2020/11/7
# ***************************************************************************************

# ＜次元削減アルゴリズム＞
# - 線形射影  ： 高次元空間から低次元空間に線形で射影を行う
#               --- 主成分分析 / 特異値分解 / ランダム射影
# - 多様体学習： 高次元空間から低次元空間に非線形で射影を行う
#               --- Isomap / 多次元尺度構成法 / 局所線形埋め込み / t-SNE / 辞書学習 / 独立成分分析


# ＜目次＞
# 3.1.1 データ準備
# 3.3 PCA(主成分分析)
# 3.3.1 PCAの概要
# 3.3.2 標準的なPCA
# 3.3.3 インクリメンタルPCA
# 3.3.4 スパースPCA
# 3.3.5 カーネルPCA
# 3.4 特異値分解
# 3.5 ランダム射影
# 3.5.1 ガウス型ランダム射影


# 3.1.1 データ準備 ----------------------------------------------------------------------

# ライブラリ準備 ********************************************************************

# Main
import gzip
import os
import pickle

import pandas as pd

# Data Prep and Model Evaluation
from sklearn import preprocessing as pp

# Data Viz
import matplotlib.pyplot as plt
import seaborn as sns

color = sns.color_palette()


# データロード ********************************************************************

# パス設定
# --- MNISTデータセット(15790kb)
current_path = os.getcwd()
file = os.path.sep.join(['', 'book', 'unsupervised', 'datasets', 'mnist_data', 'mnist.pkl.gz'])

# データ取得
# --- gzファイルにpickleファイルが保存されている
f = gzip.open(current_path + file, 'rb')
train_set, validation_set, test_set = pickle.load(f, encoding='latin1')
f.close()

# データのセット
# --- 要素[0] : numpy.ndarray(行列)
# --- 要素[1] : numpy.ndarray(ベクトル)
X_train, y_train = train_set[0], train_set[1]
X_validation, y_validation = validation_set[0], validation_set[1]
X_test, y_test = test_set[0], test_set[1]


# データセットの確認 ********************************************************************

# Train
print("Shape of X_train: ", X_train.shape)
print("Shape of y_train: ", y_train.shape)

# Validation
print("Shape of X_validation: ", X_validation.shape)
print("Shape of y_validation: ", y_validation.shape)


# Test
print("Shape of X_test: ", X_test.shape)
print("Shape of y_test: ", y_test.shape)


# Pandasデータフレームの準備 ********************************************************************

# ＜ポイント＞
# - len()で行列の行数を取得
# - DataFrameとSeriesのインデックスとして使用
# - Trainは0-49999, Validationは50000-59999, Testは60000-69999


# 訓練データ
train_index = range(0, len(X_train))
X_train = pd.DataFrame(data=X_train, index=train_index)
y_train = pd.Series(data=y_train, index=train_index)

# 検証データ
validation_index = range(len(X_train), len(X_train) + len(X_validation))
X_validation = pd.DataFrame(data=X_validation, index=validation_index)
y_validation = pd.Series(data=y_validation, index=validation_index)

# テストデータ
test_index = range(len(X_train) + len(X_validation),
                   len(X_train) + len(X_validation) + len(X_test))
X_test = pd.DataFrame(data=X_test, index=test_index)
y_test = pd.Series(data=y_test, index=test_index)


# データフレームの確認 ********************************************************************

# データ確認
X_train.head()
y_train.head()

# データ確認
X_train.describe()
y_train.describe()


# 画像の表示 *****************************************************************************

# 関数定義
# --- MNISTデータのプロット表示
def view_digit(example):
    label = y_train.loc[example]
    image = X_train.loc[example, :].values.reshape([28, 28])
    plt.title('Example: %d  Label: %d' % (example, label))
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.show()


# 画像表示
# --- 1番目
view_digit(0)


# 3.3.1 PCAの概要 ------------------------------------------------------------------

# ＜ポイント＞
# - PCAは可能な限りデータのばらつき(データの情報)を維持できる低次元表現を見つける
#   --- 相関を減らしていく過程で元の高次元データからの分散が最大の方向を見つけて低次元空間に射影
# - 次元圧縮の過程でデータサイズも圧縮される
#   --- 機械学習の前処理のパイプラインとしても使われる
# - 特徴量をスケーリングして使用する
#   --- MINSTデータは既に0-1にスケーリングされている


# ＜参考＞
# Scikit-learnの主成分分析 (PCA)
# https://helve-python.hatenablog.jp/entry/scikitlearn-pca


# 3.3.2 標準的なPCA --------------------------------------------------------------------

# Principal Component Analysis
from sklearn.decomposition import PCA


# パラメータの設定
n_components = 784
whiten = False
random_state = 2018


# インスタンスの作成
pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)


# PCAの実行
# --- numpy.ndarrayで出力
# --- 戻り値はサンプル数×n_componentsの2次元配列
X_train_PCA = pca.fit_transform(X_train)


# データフレームに変換
X_train_PCA = pd.DataFrame(data=X_train_PCA, index=train_index)


# PCAの評価
# --- 分散の合計
# --- 主成分数を784としているので全く次元削減をしていない
print("Variance Explained by all 784 principal components: ",
      sum(pca.explained_variance_ratio_))


# 分散をデータフレームに出力
importanceOfPrincipalComponents = pd.DataFrame(data=pca.explained_variance_ratio_).T


# 累積分散量
# --- PCを10個ずつ抽出
# --- 0-100で約91.5％の変動が説明されている
print('Variance Captured by First 10 Principal Components: ',
      importanceOfPrincipalComponents.loc[:, 0:9].sum(axis=1).values)
print('Variance Captured by First 20 Principal Components: ',
      importanceOfPrincipalComponents.loc[:, 0:19].sum(axis=1).values)
print('Variance Captured by First 50 Principal Components: ',
      importanceOfPrincipalComponents.loc[:, 0:49].sum(axis=1).values)
print('Variance Captured by First 100 Principal Components: ',
      importanceOfPrincipalComponents.loc[:, 0:99].sum(axis=1).values)
print('Variance Captured by First 200 Principal Components: ',
      importanceOfPrincipalComponents.loc[:, 0:199].sum(axis=1).values)
print('Variance Captured by First 300 Principal Components: ',
      importanceOfPrincipalComponents.loc[:, 0:299].sum(axis=1).values)


# プロット表示
sns.set(font_scale=1.5)
sns.barplot(data=importanceOfPrincipalComponents.loc[:, 0:9], color='k')
plt.show()


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


# プロット表示
# --- 主成分による重要度
# --- わずか2次元でも類似ラベルは近くに集まっていることが確認できる
# --- 同じ文字の画像は他の数字よりも近い位置に集まっている
scatterPlot(X_train_PCA, y_train, "PCA")


# 関数定義
# --- PC1-PC2を抽出(XY軸)
def scatterPlot_2(df_x, df_y):
    X_train_scatter = pd.DataFrame(data=df_x.loc[:, [350, 406]], index=df_x.index)
    X_train_scatter = pd.concat((X_train_scatter, df_y), axis=1, join="inner")
    X_train_scatter.columns = ["First Vector", "Second Vector", "Label"]
    sns.lmplot(x="First Vector", y="Second Vector", hue="Label", data=X_train_scatter, fit_reg=False)
    ax = plt.gca()
    ax.set_title("Separation of Observations Using Original Feature Set")
    plt.show()


# プロット表示
scatterPlot_2(X_train, y_train)


# 3.3.3 インクリメンタルPCA ------------------------------------------------------------

# ＜ポイント＞
# - メモリに乗り切らないほどの大規模データに対してはインクリメンタルPCAを使用する
#   --- バッチに分けてPCAを実行して、標準的PCAと同様の結果を得ることができる
#   --- バッチサイズは手動で決定する必要がある


# Incremental PCA
from sklearn.decomposition import IncrementalPCA


# パラメータの設定
n_components = 784
batch_size = None


# インスタンスの作成
incrementalPCA = IncrementalPCA(n_components=n_components, batch_size=batch_size)


# PCAの実行
# --- numpy.ndarrayで出力
# --- 戻り値はサンプル数×n_componentsの2次元配列
X_train_incrementalPCA = incrementalPCA.fit_transform(X_train)


# データフレームに変換
X_train_incrementalPCA = pd.DataFrame(data=X_train_incrementalPCA, index=train_index)


# プロット表示
scatterPlot(X_train_incrementalPCA, y_train, "Incremental PCA")


# 3.3.4 スパースPCA ------------------------------------------------------------

# ＜ポイント＞
# - 行準的PCAは元の特徴量空間を可能な限り密に表現しようとする
# - スパースPCAは、スパース性を残しながらPCAを解く手法

# ＜参考＞
# スパース主成分分析
# - https://stats.biopapyrus.jp/sparse-modeling/sparse-pca.html


# ＜スパースとは＞
# - スパースは、｢まばらな｣を意味する
# - スパース性とは、物事の本質的な特徴を決定づける要素はわずかであるという性質を示す


# Sparse PCA
from sklearn.decomposition import SparsePCA


# パラメータの設定
n_components = 100
alpha = 0.0001
random_state = 2018
n_jobs = -1


# インスタンスの作成
sparsePCA = SparsePCA(n_components=n_components,
                      alpha=alpha, random_state=random_state, n_jobs=n_jobs)


# PCAの実行
# --- 計算に時間がかかるので最初の10000行のみ使用
sparsePCA.fit(X_train.loc[:10000, :])
X_train_sparsePCA = sparsePCA.transform(X_train)


# データフレームに変換
X_train_sparsePCA = pd.DataFrame(data=X_train_sparsePCA, index=train_index)

# プロット表示
scatterPlot(X_train_sparsePCA, y_train, "Sparse PCA")


# 3.3.5 カーネルPCA ------------------------------------------------------------

# ＜ポイント＞
# - 元の低次元空間を圧縮する際に非線形に射影する手法
#   --- 類似度関数(カーネル法)を用いて行われる
#   --- 元の特徴量空間が線形分離できない場合に有効性が高い
# - 一般的によく使われるアルゴリズムとしてRBFカーネルというものがある
#   --- RBF：Radial Basis Function
#   --- 動径分布関数カーネル

# Kernel PCA
from sklearn.decomposition import KernelPCA


# パラメータの設定
n_components = 100
kernel = 'rbf'
gamma = None
random_state = 2018
n_jobs = 1


# インスタンスの作成
kernelPCA = KernelPCA(n_components=n_components, kernel=kernel,
                      gamma=gamma, n_jobs=n_jobs, random_state=random_state)


# PCAの実行
# --- 計算に時間がかかるので最初の10000行のみ使用
kernelPCA.fit(X_train.loc[:10000, :])
X_train_kernelPCA = kernelPCA.transform(X_train)


# データフレームに変換
X_train_kernelPCA = pd.DataFrame(data=X_train_kernelPCA, index=train_index)


# プロット表示
scatterPlot(X_train_kernelPCA, y_train, "Kernel PCA")


# 3.4 特異値分解 ------------------------------------------------------------

# ＜ポイント＞
# - 元の特徴量行列のランクよりも小さいランクを持つ行列を作り、2つの行列を線形結合することで元の行列を再編成する手法
#   --- 特異値分解(SVD: Singular Value Decomposition)
#   --- 小さい行列を作る際に、元の行列の最も多くの情報を持つベクトルを維持する
#   --- 小さいランクの行列は、元の特徴量空間の最も重要な要素を捉えている


# Singular Value Decomposition
from sklearn.decomposition import TruncatedSVD


# パラメータの設定
n_components = 200
algorithm = 'randomized'
n_iter = 5
random_state = 2018


# インスタンスの作成
svd = TruncatedSVD(n_components=n_components, algorithm=algorithm,
                   n_iter=n_iter, random_state=random_state)


# SVDの実行
X_train_svd = svd.fit_transform(X_train)


# データフレームに変換
X_train_svd = pd.DataFrame(data=X_train_svd, index=train_index)


# プロット表示
scatterPlot(X_train_svd, y_train, "Singular Value Decomposition")


# 3.5 ランダム射影 ------------------------------------------------------------

# ＜ポイント＞
# - Johnson-Lindenstraussの補題に基づく手法
#   --- 高次元空間をはるかに低次元な空間に埋め込んだ場合に、点間の距離がほぼ保存されるというもの
#   --- 高次元空間から低次元空間に移しても元の特徴量の構造は保存されることを意味する
# - ランダム射影には以下の2つの方法がある
#   --- ガウス型ランダム射影
#   --- スパースランダム射影


# 3.5.1 ガウス型ランダム射影 ------------------------------------------------------------

# ＜ポイント＞


# Gaussian Random Projection
from sklearn.random_projection import GaussianRandomProjection


# パラメータの設定
n_components = 'auto'
eps = 0.5
random_state = 2018


# インスタンスの作成
GRP = GaussianRandomProjection(n_components=n_components, eps=eps,
                               random_state=random_state)


# ガウス型ランダム射影の実行
X_train_GRP = GRP.fit_transform(X_train)


# データフレームに変換
X_train_GRP = pd.DataFrame(data=X_train_GRP, index=train_index)


# プロット表示
scatterPlot(X_train_GRP, y_train, "Gaussian Random Projection")


# 3.5.2 スパースランダム射影 ------------------------------------------------------------

# Sparse Random Projection
from sklearn.random_projection import SparseRandomProjection


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


# ガウス型ランダム射影の実行
X_train_SRP = SRP.fit_transform(X_train)


# データフレームに変換
X_train_SRP = pd.DataFrame(data=X_train_SRP, index=train_index)


# プロット表示
scatterPlot(X_train_SRP, y_train, "Sparse Random Projection")


# 3.6 Isomap ------------------------------------------------------------

# ＜ポイント＞
# - Isomapは非線形に次元削減を行う手法(多様体学習)の最も基本的な手法
#   --- Isomap： Isometric Mapping
# - 全ての距離は曲線距離(curved distance)もしくは測地線距離(geodesic distance)で計算する
#   --- ユーグリッド距離のような直線距離ではない
#   --- 元の特徴量集合の低次元空間への新たな埋め込みを学習する
#   --- 元データに内在する幾何構造を学習する


# Isomap
from sklearn.manifold import Isomap


# パラメータの設定
n_neighbors = 5
n_components = 10
n_jobs = 4


# インスタンスの作成
isomap = Isomap(n_neighbors=n_neighbors,
                n_components=n_components, n_jobs=n_jobs)


# Isomapの実行
isomap.fit(X_train.loc[0:5000, :])
X_train_isomap = isomap.transform(X_train)


# データフレームに変換
X_train_isomap = pd.DataFrame(data=X_train_isomap, index=train_index)


# プロット表示
scatterPlot(X_train_isomap, y_train, "Isomap")


# 3.7 MDS(多次元尺度構成法) ------------------------------------------------------------

# ＜ポイント＞
# - MDSは元のデータセット中のデータポイント間の類似度を学習し、その類似度を用いて低次元空間にモデル化する
#   --- MDS： Multi-Dimensional Scaling


# Multidimensional Scaling
from sklearn.manifold import MDS


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


# NDSの実行
X_train_mds = mds.fit_transform(X_train.loc[0:1000, :])


# データフレームに変換
X_train_mds = pd.DataFrame(data=X_train_mds, index=train_index[0:1001])


# プロット表示
scatterPlot(X_train_mds, y_train, "Multidimensional Scaling")


# 3.8 LLE(局所線形埋め込み) ------------------------------------------------------------

# ＜ポイント＞
# - 元の特徴量空間から低次元空間に移す際に、局所的な近傍での距離を保つように射影する
#   --- LLE： Locally Linear Embedding
#   --- 小さい成分に分割して線形埋め込みとしてモデル化することで、元の高次元データにある非線形な構造を見つけ出す


# Locally Linear Embedding (LLE)
from sklearn.manifold import LocallyLinearEmbedding


# パラメータの設定
n_neighbors = 10
n_components = 2
method = 'modified'
n_jobs = 4
random_state = 2018


# インスタンスの作成
lle = LocallyLinearEmbedding(n_neighbors=n_neighbors,
                             n_components=n_components, method=method,
                             random_state=random_state, n_jobs=n_jobs)


# LLEの実行
lle.fit(X_train.loc[0:5000, :])
X_train_lle = lle.transform(X_train)
X_train_lle = pd.DataFrame(data=X_train_lle, index=train_index)


# プロット表示
scatterPlot(X_train_lle, y_train, "Locally Linear Embedding")


# 3.9 t-SNE ------------------------------------------------------------

# ＜ポイント＞
# -


# t-SNE
from sklearn.manifold import TSNE


# パラメータの設定
n_components = 2
learning_rate = 300
perplexity = 30
early_exaggeration = 12
init = 'random'
random_state = 2018


# インスタンスの作成
tSNE = TSNE(n_components=n_components, learning_rate=learning_rate,
            perplexity=perplexity, early_exaggeration=early_exaggeration,
            init=init, random_state=random_state)


# t-SNEの実行
X_train_tSNE = tSNE.fit_transform(X_train_PCA.loc[:5000, :9])
X_train_tSNE = pd.DataFrame(data=X_train_tSNE, index=train_index[:5001])


# プロット表示
scatterPlot(X_train_tSNE, y_train, "t-SNE")


# 3.11 辞書学習 ------------------------------------------------------------

# ＜ポイント＞
# -


# Mini-batch dictionary learning
from sklearn.decomposition import MiniBatchDictionaryLearning


# パラメータの設定
n_components = 50
alpha = 1
batch_size = 200
n_iter = 25
random_state = 2018


# インスタンスの作成
miniBatchDictLearning = MiniBatchDictionaryLearning(
    n_components=n_components, alpha=alpha,
    batch_size=batch_size, n_iter=n_iter,
    random_state=random_state)


# 辞書学習の実行
miniBatchDictLearning.fit(X_train.loc[:, :10000])
X_train_miniBatchDictLearning = miniBatchDictLearning.fit_transform(X_train)
X_train_miniBatchDictLearning = pd.DataFrame(
    data=X_train_miniBatchDictLearning, index=train_index)


# プロット表示
scatterPlot(X_train_miniBatchDictLearning, y_train,
            "Mini-batch Dictionary Learning")

# 3.12 ICA(独立性分分析) ------------------------------------------------------------

# ＜ポイント＞
# -　


# Independent Component Analysis
from sklearn.decomposition import FastICA


# パラメータの設定
n_components = 25
algorithm = 'parallel'
whiten = True
max_iter = 100
random_state = 2018


# インスタンスの作成
fastICA = FastICA(n_components=n_components, algorithm=algorithm,
                  whiten=whiten, max_iter=max_iter, random_state=random_state)


# ICAの実行
X_train_fastICA = fastICA.fit_transform(X_train)
X_train_fastICA = pd.DataFrame(data=X_train_fastICA, index=train_index)


# プロット表示
scatterPlot(X_train_fastICA, y_train, "Independent Component Analysis")
