# ******************************************************************************
# Chapter   : 1 機械学習の枠組みを理解する
# Title     : 1.6 Numpyとmatplotlibを使ってプロット(Recipe5)
# Created by: Owner
# Created on: 2020/12/20
# Page      : P16 - P21
# ******************************************************************************

# ＜概要＞
# - matplotlibはpythonの主要なチャートライブラリ


# ＜目次＞
# 0 準備
# 1 matplotlibの基本操作
# 2 irisのプロット


# 0 準備 -------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets

# データロード
iris = datasets.load_iris()

# データフレーム定義
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df


# 1 matplotlibの基本操作 ----------------------------------------------------------------------------

# ＜ポイント＞
# - サブプロットの最初の2つの数字でグリッドを指定、最後の数字で位置を指定


# matplotlibでプロット
# --- 直線
plt.plot(np.arange(10), np.arange(10))
plt.show()

# 指数関数のプロット
plt.plot(np.arange(10), np.exp(np.arange(10)))
plt.show()

# プロットを2つ並べる
# --- 左右に並べる
plt.figure()
plt.subplot(121)
plt.plot(np.arange(10), np.exp(np.arange(10)))
plt.subplot(122)
plt.scatter(np.arange(10), np.exp(np.arange(10)))
plt.show()

# プロットを2つ並べる
# --- 上下に並べる
plt.figure()
plt.subplot(211)
plt.plot(np.arange(10), np.exp(np.arange(10)))
plt.subplot(212)
plt.scatter(np.arange(10), np.exp(np.arange(10)))
plt.show()

# 2*2のプロット
# --- 行ごとに列順に表示（表示される順序に注意）
plt.figure()
plt.subplot(221)
plt.plot(np.arange(10), np.exp(np.arange(10)))
plt.subplot(222)
plt.scatter(np.arange(10), np.exp(np.arange(10)))
plt.subplot(223)
plt.plot(np.arange(10), np.exp(np.arange(10)))
plt.subplot(224)
plt.scatter(np.arange(10), np.exp(np.arange(10)))
plt.show()


# 2 irisのプロット --------------------------------------------------------------------------------

# データ準備
iris = datasets.load_iris()
data = iris.data
target = iris.target

# プロット作成
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(data[:, 0], data[:, 1], c=target)
plt.subplot(122)
plt.scatter(data[:, 2], data[:, 3], c=target)
plt.show()
