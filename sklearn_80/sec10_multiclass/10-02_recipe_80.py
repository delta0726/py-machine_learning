# ******************************************************************************
# Chapter   : 10 テキスト分類と多クラス分類
# Title     : 10-2 ナイーブベイズを使って文書を分類する（Recipe80)
# Created by: Owner
# Created on: 2020/12/30
# Page      : P308 - P313
# ******************************************************************************

# ＜概要＞
# -


# ＜目次＞
# 0 準備
# 1 テキストの前処理
# 2 データ準備


# 0 準備 ------------------------------------------------------------------------------------------

import numpy as np


from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score


# データロード
categories = ["rec.autos", "rec.motorcycles"]
newgroups = fetch_20newsgroups(categories=categories)

# データ確認
print("\n".join(newgroups.data[:1]))

# カテゴリ確認
newgroups.target_names


# 1 テキストの前処理 ----------------------------------------------------------------------------

# Bow行列に変換
count_vec = CountVectorizer()
bow = count_vec.fit_transform(newgroups.data)

# Bow行列の確認
# --- 疎行列になっている
bow

bow = np.array(bow.todense())

words = np.array(count_vec.get_feature_names())
words[bow[0] > 0][:5]

'10pm' in newgroups.data[0].lower()

'1qh336innfl5' in newgroups.data[0].lower()


# 2 データ準備 ------------------------------------------------------------------------------

# デー格納
X = bow
y = newgroups.target

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y)


# 3 モデリング ------------------------------------------------------------------------------

# インスタンス生成
clf = naive_bayes.GaussianNB()

# 学習
clf.fit(X_train, y_train)


# 4 モデル評価 ------------------------------------------------------------------------------

# 予測
y_pred = clf.predict(X_test)

# Accuracy
accuracy_score(y_true=y_test, y_pred=y_pred)

