# ******************************************************************************
# Title     : Scikit-Learnデータ分析実装ハンドブック
# Chapter   : 6 次元削減アルゴリズム
# Theme     : 6-3 カーネルPCA
# Created by: Owner
# Created on: 2021/5/22
# Page      : P278 - P282
# ******************************************************************************

# ＜概要＞
# - カーネルPCAは高次元特徴量で主成分分析するので、非線形なデータに対しても適用できる


# ＜目次＞
# 0 準備
# 1 前処理
# 2 カーネルPCAの実行
# 3 カーネル主成分を軸としたロジスティック回帰
# 4 プロット


# 0 準備 ------------------------------------------------------------------------------

# ライブラリ
from pprint import pprint
import matplotlib.pyplot as plt

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import load_wine
from sklearn.decomposition import KernelPCA


# データロード
wine = load_wine()

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


# 2 カーネルPCAの実行 ------------------------------------------------------------------

# モデル構築
KPCA = KernelPCA(n_components=2, kernel='rbf', gamma=0.3)

# モデル訓練
X_train_kpca = KPCA.fit_transform(X_train_std)

# 確認
pprint(vars(KPCA))

# モデル適用
X_test_kpca = KPCA.transform(X_test_std)


# 3 カーネル主成分を軸としたロジスティック回帰 ---------------------------------------------

# ＜ポイント＞
# - カーネルPCAで前処理することで、ロジスティック回帰による線形分類が可能となる


# モデル構築
model = LogisticRegression(multi_class='ovr', max_iter=100, solver='liblinear',
                           penalty='l2', random_state=0)

# モデル訓練
model.fit(X=X_train_kpca, y=y_train)

# 確認
pprint(vars(model))

# 予測
y_test_pred = model.predict(X_test_kpca)

# 正解率
accuracy_score(y_true=y_test, y_pred=y_test_pred)


# 4 プロット ---------------------------------------------------------------------------

# 訓練データのプロット
plt.figure(figsize=(8,4))
plot_decision_regions(X_train_kpca, y_train, model)
plt.show()

# テストデータのプロット
plt.figure(figsize=(8,4))
plot_decision_regions(X_test_kpca, y_test, model)
plt.show()
