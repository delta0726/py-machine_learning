# ******************************************************************************
# Chapter   : 1 機械学習の枠組みを理解する
# Title     : 1.5 irisデータセットをpandasで可視化(Recipe4)
# Created by: Owner
# Created on: 2020/12/20
# Page      : P14 - P16
# ******************************************************************************

# ＜概要＞
# - pandasはmatplotlibをベースとした簡単なチャート機能を備える


# ＜目次＞
# 0 準備
# 1 pandasによる可視化


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


# 1 pandasによる可視化 ---------------------------------------------------------

# ヒストグラムの作成
# --- sepal_length
iris_df['sepal length (cm)'].hist(bins=30)
plt.show()

# ヒストグラムの作成
# --- グループで色分け（speciesより取得）
# --- ループごとに50個の要素を取得してヒストグラムを重ね書き
# --- np.where(iris.target == class_number)[0]の[0]ななくても動作する
# class_number = 0
for class_number in np.unique(iris.target):
    plt.figure(1)
    iris_df['sepal length (cm)'].iloc[np.where(iris.target == class_number)[0]].hist(bins=30)
plt.show()
