# ***************************************************************************************
# Book      : Pythonではじめる教師なし学習
# Chapter   : 3.次元削減
# Title     : 3.3 PCA(主成分分析)
# Created by: Owner
# Created on: 2020/12/31
# Page      : P72 - P80
# ***************************************************************************************


# ＜PCAの概要＞
# - PCAは可能な限りデータのばらつき(データの情報)を維持できる低次元表現を見つける
#   --- 相関を減らしていく過程で元の高次元データからの分散が最大の方向を見つけて低次元空間に射影する
#   ---- 次元圧縮の過程でデータ量も圧縮される（メモリ削減効果）
#   --- 機械学習の前処理のパイプラインとしても使われる


# ＜目次＞
# 0 準備
# 1 標準的なPCA
# 2 PCAの評価
# 3 PCAの可視化
# 4 インクリメンタルPCA
# 5 スパースPCA
# 6 カーネルPCA


# 0 準備 --------------------------------------------------------------------------------

# Main
import gzip
import os
import pickle
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import SparsePCA


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


# 1 標準的なPCA --------------------------------------------------------------------

# ＜PCAの概要＞
# - PCAは可能な限りデータのばらつき(データの情報)を維持できる低次元表現を見つける
#   --- 相関を減らしていく過程で元の高次元データからの分散が最大の方向を見つけて低次元空間に射影する
#   ---- 次元圧縮の過程でデータサイズも圧縮される
#   --- 機械学習の前処理のパイプラインとしても使われる
# - 特徴量をスケーリングして使用する
#   --- MNISTデータは既に0-1にスケーリングされている


# ＜参考＞
# Scikit-learnの主成分分析 (PCA)
# https://helve-python.hatenablog.jp/entry/scikitlearn-pca


# パラメータの設定
n_components = 784
whiten = False
random_state = 2018

# インスタンスの作成
pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
vars(pca)

# 学習器の作成
pca.fit(X_train)

# データ適用
# --- 戻り値は次元圧縮後のデータ
X_train_PCA = pca.transform(X_train)

# データフレームに変換
X_train_PCA = pd.DataFrame(data=X_train_PCA, index=train_index)

# データ確認
# --- 元データ（列名はピクセル）
# --- PCA適用後のデータ（列名は次元NO）
X_train.head()
X_train_PCA.head()

# 行列数
X_train_PCA.shape


# 2 PCAの評価 -----------------------------------------------------------------------

# 分散の合計
# --- 主成分数を784としているので全く次元削減がされていない
sum(pca.explained_variance_ratio_)

# 分散をデータフレームに出力
importanceOfPrincipalComponents = pd.DataFrame(data=pca.explained_variance_ratio_).T

# 累積分散量
# --- PCを10個ずつ抽出
# --- 0-100で約91.5％の変動が説明されている
importanceOfPrincipalComponents.loc[:, 0:9].sum(axis=1)
importanceOfPrincipalComponents.loc[:, 0:19].sum(axis=1)
importanceOfPrincipalComponents.loc[:, 0:49].sum(axis=1)
importanceOfPrincipalComponents.loc[:, 0:99].sum(axis=1)
importanceOfPrincipalComponents.loc[:, 0:199].sum(axis=1)
importanceOfPrincipalComponents.loc[:, 0:299].sum(axis=1)

# プロット表示
# --- PC1-PC10の分散量を表示
sns.set(font_scale=1.5)
sns.barplot(data=importanceOfPrincipalComponents.loc[:, 0:9], color='k')
plt.show()


# 3 PCAの可視化 -----------------------------------------------------------------------

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


# 4 インクリメンタルPCA ------------------------------------------------------------

# ＜ポイント＞
# - メモリに乗り切らないほどの大規模データに対してはインクリメンタルPCAを使用する
#   --- バッチに分けてPCAを実行して、標準的PCAと同様の結果を得ることができる
#   --- バッチサイズは手動で決定する必要がある


# パラメータの設定
n_components = 784
batch_size = None

# インスタンス生成
incrementalPCA = IncrementalPCA(n_components=n_components, batch_size=batch_size)
vars(incrementalPCA)

# 学習器の作成
incrementalPCA.fit(X_train)

# 学習器の適用
X_train_incrementalPCA = incrementalPCA.transform(X_train)

# データフレームに変換
X_train_incrementalPCA = pd.DataFrame(data=X_train_incrementalPCA, index=train_index)

# プロット表示
scatterPlot(X_train_incrementalPCA, y_train, "Incremental PCA")


# 5 スパースPCA --------------------------------------------------------------------------

# ＜ポイント＞
# - 行準的PCAは元の特徴量空間を可能な限り密に表現しようとする
# - スパースPCAは、スパース性を残しながらPCAを解く手法

# ＜参考＞
# スパース主成分分析
# - https://stats.biopapyrus.jp/sparse-modeling/sparse-pca.html


# ＜スパースとは＞
# - スパースは、｢まばらな｣を意味する
# - スパース性とは、物事の本質的な特徴を決定づける要素はわずかであるという性質を示す

# パラメータの設定
n_components = 100
alpha = 0.0001
random_state = 2018
n_jobs = -1

# インスタンス生成
sparsePCA = SparsePCA(n_components=n_components,
                      alpha=alpha, random_state=random_state, n_jobs=n_jobs)

# 学習器の作成
# --- 計算に時間がかかるので最初の10000行のみ使用
sparsePCA.fit(X_train.loc[:10000, :])

# 学習器の適用
X_train_sparsePCA = sparsePCA.transform(X_train)

# データフレームに変換
X_train_sparsePCA = pd.DataFrame(data=X_train_sparsePCA, index=train_index)

# プロット表示
scatterPlot(X_train_sparsePCA, y_train, "Sparse PCA")


# 6 カーネルPCA ------------------------------------------------------------

# ＜ポイント＞
# - 元の低次元空間を圧縮する際に非線形に射影する手法
#   --- 類似度関数(カーネル法)を用いて行われる
#   --- 元の特徴量空間が線形分離できない場合に有効性が高い
# - 一般的によく使われるアルゴリズムとしてRBFカーネルというものがある
#   --- RBF：Radial Basis Function
#   --- 動径分布関数カーネル

# パラメータの設定
n_components = 100
kernel = 'rbf'
gamma = None
random_state = 2018
n_jobs = 1

# インスタンスの作成
kernelPCA = KernelPCA(n_components=n_components, kernel=kernel,
                      gamma=gamma, n_jobs=n_jobs, random_state=random_state)


# 学習器の作成
# --- 計算に時間がかかるので最初の10000行のみ使用
kernelPCA.fit(X_train.loc[:10000, :])

# 学習器の適用
X_train_kernelPCA = kernelPCA.transform(X_train)


# データフレームに変換
X_train_kernelPCA = pd.DataFrame(data=X_train_kernelPCA, index=train_index)


# プロット表示
scatterPlot(X_train_kernelPCA, y_train, "Kernel PCA")
