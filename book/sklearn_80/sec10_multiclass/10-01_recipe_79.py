# ******************************************************************************
# Chapter   : 10 テキスト分類と多クラス分類
# Title     : 10-1 分類に確率的勾配降下法を使用する（Recipe79)
# Created by: Owner
# Created on: 2020/12/30
# Page      : P307 - P308
# ******************************************************************************

# ＜概要＞
# - 確率的勾配降下法(SGD)は回帰モデルを適合させるための基本的な手法


# ＜目次＞
# 0 準備
# 1 モデリング
# 2 モデル評価


# 0 準備 ------------------------------------------------------------------------------------------

from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# データロード
X, y = datasets.make_classification(n_samples=500)

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

# ラベル確認
y_train[:10]


# 1 モデリング -----------------------------------------------------------------------------------

# インスタンス生成
# --- 確率的勾配降下法(SGD)分類モデル
sgd_clf = linear_model.SGDClassifier()

# 学習
sgd_clf.fit(X_train, y_train)


# 2 モデル評価 -----------------------------------------------------------------------------------

# 予測
y_pred = sgd_clf.predict(X_test)

# Accuracy
accuracy_score(y_true=y_test, y_pred=y_pred)
