# ******************************************************************************
# Title     : Scikit-Learnデータ分析実装ハンドブック
# Chapter   : 6 次元削減アルゴリズム
# Theme     : 6-2 主成分分析
# Created by: Owner
# Created on: 2021/5/22
# Page      : P267 - P275
# ******************************************************************************


# ＜概要＞
# - 線形の次元圧縮の手法としてPCAを確認する
# - {sklearn}のPCAでは以下の3パターンの指定が可能
#   --- kを整数で指定 ：指定した次元数に圧縮する
#   --- kをNone     ：元データと同じ次元数にする
#   --- kを小数で指定 ：累積寄与率が指定する水準に到達するPC数に圧縮


# ＜目次＞
# 0 準備
# 1 前処理
# 2 主成分分析
# 3 PCAの要素を確認
# 4 ロジスティック回帰
# 5 プロット作成
# 6 次元数を指定しないPCA
# 7 累積寄与率を指定したPCA


# 0 準備 ------------------------------------------------------------------------------

    # ライブラリ
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# データ準備
wine = datasets.load_wine()

# データ格納
X = wine.data
y = wine.target


# 1 前処理 ---------------------------------------------------------------------------

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# データ基準化
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

# データ形状
# --- (142, 13)
# --- (36, 13)
X_train_std.shape
X_test_std.shape

# データ確認
X_train_std.mean(axis=0)
X_train_std.std(axis=0)


# 2 主成分分析 ------------------------------------------------------------------------

# インスタンス生成
PCA = PCA(n_components=2)

# モデル構築＆学習
X_train_pca = PCA.fit_transform(X=X_train_std)

# 確認
pprint(vars(PCA))

# モデル適用
X_test_pca = PCA.transform(X_test_std)


# 3 PCAの要素を確認 ------------------------------------------------------------------------

# 固有値
# --- 説明される分散量
PCA.explained_variance_

# 因子寄与率
# --- 説明される分散量 / 全体の分散量
PCA.explained_variance_ratio_

# 固有ベクトル
# --- (2, 13)
PCA.components_
PCA.components_.shape

# 次元圧縮されたデータ
# --- 2次元
X_train_pca
X_train_pca.shape
X_test_pca
X_test_pca.shape


# 4 ロジスティック回帰 ----------------------------------------------------------------------

# モデル構築
model = LogisticRegression(multi_class='ovr', max_iter=100, solver='liblinear',
                           penalty='l2', random_state=0)

# モデル訓練
model.fit(X=X_train_pca, y=y_train)

# 確認
pprint(vars(model))

# 予測
y_test_pred = model.predict(X_test_pca)

# 正解率
accuracy_score(y_true=y_test, y_pred=y_test_pred)


# 5 プロット作成 -------------------------------------------------------------------------

# 訓練データのプロット
plt.figure(figsize=(8,4))
plot_decision_regions(X_train_pca, y_train, model)
plt.show()

# テストデータのプロット
plt.figure(figsize=(8,4))
plot_decision_regions(X_test_pca, y_test, model)
plt.show()


# 6 次元数を指定しないPCA ------------------------------------------------------------------

# モデル構築
# --- 要素数を指定しない
PCA2= PCA(n_components=None)

# PCAの実行
PCA2.fit_transform(X_train_std)

# 確認
pprint(vars(PCA2))

# 出力確認
# --- 固有値
# --- 因子寄与率
PCA2.explained_variance_
PCA2.explained_variance_ratio_

# プロット
# --- 累積寄与率の計算
# --- プロットのサイズ指定
# --- ラインチャートの作成
ratio = PCA2.explained_variance_ratio_
ratio = np.hstack([0, ratio.cumsum()])

plt.figure(figsize=(8,4))
plt.plot(ratio)
plt.ylabel('Cumulative contribution rate')
plt.xlabel('Principal component index k')
plt.title('Wine dataset')
plt.show()


# 7 累積寄与率を指定したPCA ------------------------------------------------------------------

# モデル構築
# --- 要素数を小数で指定
PCA3= PCA(n_components=0.8)

# PCAの実行
PCA3.fit_transform(X_train_std)

# 確認
pprint(vars(PCA3))

# データ確認
# --- 固有値(説明される分散量)
# --- 因子寄与率(固有値の寄与率)
# --- 因子累積寄与率が0.8をすぎるところまでPCが作成される
PCA3.explained_variance_
PCA3.explained_variance_ratio_
PCA3.explained_variance_ratio_.cumsum()
