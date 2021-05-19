# ******************************************************************************
# Title     : Scikit-Learnデータ分析実装ハンドブック
# Chapter   : 5 クラスタリング
# Theme     : 5-3-1 混合ガウス分布(GMM)によるワインデータのクラスタリングの実装
# Created by: Owner
# Created on: 2021/5/18
# Page      : P245 - P250
# ******************************************************************************


# ＜概要＞
# - 混合ガウス分布を用いてもでるを想定してクラスタリングを行う
# - 共分散行列の対角成分をdiag/fullで指定する
#   --- diag： 中心から上下に楕円を想定する(P241)
#   --- full： 中心から斜めに楕円を想定する(P241)


# ＜目次＞
# 0 準備
# 1 データ基準化
# 2 モデル構築
# 3 クラスタ数ごとの可視化
# 4 クラスタリングの予測


# 0 準備 -------------------------------------------------------------------------------------

# ライブラリ
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# データロード
wine = load_wine()

# データ格納
X = wine.data[:, [9, 12]]
y = wine.target


# 1 データ基準化 ------------------------------------------------------------------------------

# インスタンス生成
sc = StandardScaler()

# データ変換
X_std = sc.fit_transform(X)

# データ確認
np.round(X_std.mean(axis=0))
np.round(X_std.std(axis=0))


# 2 モデル構築 --------------------------------------------------------------------------------

# インスタンス生成
# --- covariance_type='diag'：非対角成分がゼロのガウス分布を想定
model1 = GaussianMixture(n_components=3, covariance_type='diag', random_state=1)
model2 = GaussianMixture(n_components=3, covariance_type='full', random_state=1)

# モデル訓練
model1.fit(X_std)
model2.fit(X_std)

# データ確認
pprint(vars(model1))
pprint(vars(model2))


# 3 クラスタ数ごとの可視化 -------------------------------------------------------------------------

#プロットのサイズ指定
plt.figure(figsize=(8,8))

# モデル1: GMM(diag)
# --- 色とプロリンの散布図のクラスタリング
plt.subplot(2, 1, 1)

x = np.linspace(X_std[:,0].min(), X_std[:,0].max(), 100)
y = np.linspace(X_std[:,0].min(), X_std[:,0].max(), 100)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -model1.score_samples(XX)
Z = Z.reshape(X.shape)

plt.contour(X, Y, Z, levels=[0.5, 1, 2 ,3 ,4, 5]) # 等高線のプロット
plt.scatter(X_std[:,0], X_std[:,1], c=model1.predict(X_std))
plt.scatter(model1.means_[:,0], model1.means_[:,1],s=250, marker='*',c='red')
plt.title('GMM(covariance_type=diag)')


# モデル2: GMM(full)
# --- 色とプロリンの散布図のクラスタリング
plt.subplot(2, 1, 2)

x = np.linspace(X_std[:,0].min(), X_std[:,0].max(), 100)
y = np.linspace(X_std[:,0].min(), X_std[:,0].max(), 100)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -model2.score_samples(XX)
Z = Z.reshape(X.shape)

plt.contour(X, Y, Z, levels=[0.5, 1, 2 ,3 ,4, 5]) # 等高線のプロット
plt.scatter(X_std[:,0], X_std[:,1], c=model2.predict(X_std))
plt.scatter(model2.means_[:,0], model2.means_[:,1],s=250, marker='*',c='red')
plt.title('GMM(covariance_type=full)')

plt.show()


# 4 クラスタリングの予測 -------------------------------------------------------------------------

# 予測
model1.predict(X_std)
model2.predict(X_std)

# 混合係数
# --- ウエイトなので合計1となる
model1.weights_
model2.weights_

# 平均ベクトル
# --- 各クラスタの座標
model1.means_
model2.means_

# 共分散
# --- model1(diag)は非対角成分がゼロのため、対角成分のみが表示される
# --- model2(full)は非対角成分が出力されるため、行列の非対角成分も表示される
model1.covariances_
model2.covariances_
