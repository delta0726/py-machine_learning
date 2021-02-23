# ******************************************************************************
# Chapter   : 10 テキスト分類と多クラス分類
# Title     : 10-3 半教師あり学習によるラベル伝播法（Recipe81)
# Created by: Owner
# Created on: 2020/12/30
# Page      : P313 - P316
# ******************************************************************************

# ＜概要＞
# - ラベル伝播法は、ラベル付けされたデータとされていないデータを使って、ラベル付けされていないデータを学習する
#   --- 半教師あり学習の1つ


# ＜目次＞
# 0 準備


# 0 準備 ------------------------------------------------------------------------------------------

import numpy as np

from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation, LabelSpreading


# データロード
iris = datasets.load_iris()

# データ格納
X = iris.data.copy()
y = iris.target.copy()

# ラベル格納
# --- irisの3種類のカテゴリに｢unlabeled｣を加える
names = iris.target_names.copy()
names = np.append(names, ['unlabeled'])
names

# ラベルなしデータの作成
y[np.random.choice([True, False], len(y))] = -1
y[:10]
names[y[:10]]


# 1 モデリング ----------------------------------------------------------------------------

# インスタンス生成
lp = LabelPropagation()

# 学習
lp.fit(X, y)


# 2 モデル評価 ---------------------------------------------------------------------------

# 予測
y_pred = lp.predict(X)

# 正解率
(y_pred == iris.target).mean()


# 3 Label Spreadingによる学習 -----------------------------------------------------------

# インスタンス生成
ls = LabelSpreading()

# 学習
ls.fit(X, y)

# 予測
y_pred = ls.predict(X)

# 正解率
(y_pred == iris.target).mean()
