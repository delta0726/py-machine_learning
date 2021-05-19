# ******************************************************************************
# Title     : Scikit-Learnデータ分析実装ハンドブック
# Chapter   : 5 クラスタリング
# Theme     : 5-3-2 変分混合ガウス分布(VBGMM)によるワインデータのクラスタリングの実装
# Created by: Owner
# Created on: 2021/5/18
# Page      : P251 - P255
# ******************************************************************************


# ＜概要＞
# - 変分混合ガウス分布(VBGMM)のアルゴリズムは混合ガウス分布を用いてデータをクラスタリング
#   --- クラスタ数は予測モデルが自動提案


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
from sklearn.mixture import BayesianGaussianMixture
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
model3 = BayesianGaussianMixture(n_components=10, covariance_type='diag', random_state=1)

# 学習
model3.fit(X_std)

# 確認
pprint(vars(model3))


# 3 クラスタ数ごとの可視化 -------------------------------------------------------------------------

#プロットのサイズ指定
plt.figure(figsize=(8, 4))

# 色とプロリンの散布図のVBGMMによるクラスタリング
x = np.linspace(X_std[:,0].min(), X_std[:,0].max(), 100)
y = np.linspace(X_std[:,0].min(), X_std[:,0].max(), 100)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -model3.score_samples(XX)
Z = Z.reshape(X.shape)

plt.contour(X, Y, Z, levels=[0.5, 1, 2 ,3 ,4, 5]) # 等高線のプロット
plt.scatter(X_std[:,0], X_std[:,1], c=model3.predict(X_std))
plt.title('VBGMM(covariance_type=full)')

plt.show()


# 4 クラスタリングの予測 -------------------------------------------------------------------------

# 予測
model3.predict(X_std)

# 混合係数
# --- ウエイトなので合計1となる
model3.weights_

# 混合係数の可視化
x =np.arange(1, model3.n_components+1)
plt.figure(figsize=(8,4)) #プロットのサイズ指定
plt.bar(x, model3.weights_, width=0.7, tick_label=x)

plt.ylabel('Mixing weights for each mixture component')
plt.xlabel('Number of mixture components')
plt.title('Wine dataset')
plt.show()
