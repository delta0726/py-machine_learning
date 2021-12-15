# ******************************************************************************
# Title       : Scikit-Learnデータ分析実装ハンドブック
# Chapter     : 2 Scikit-Learnと開発環境
# Theme       : 2-3 機械学習の基本的な実装
# Creat Date  : 2021/5/13
# Final Update: 2021/11/13
# Page        : P42 - P45
# ******************************************************************************


# ＜概要＞
# - 機械学習の基本フローを確認する
#   --- 外部からデータをインポート
#   --- 決定木によるマルチクラス問題


# ＜目次＞
# 0 準備
# 1 訓練データと評価データの準備
# 2 アルゴリズムの選択
# 3 学習
# 4 予測


# 0 準備 -------------------------------------------------------------------------------------

# ライブラリ
import pprint

import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# データロード
npArray = np.loadtxt("data/in.csv", delimiter=",", dtype="float", skiprows=1)

# データ準備
# --- x：説明変数
# --- y：目的変数（マルチクラスのラベル：1, 2, 3）
x = npArray[:, 1:3]
y = npArray[:, 3:4]

# 元データの確認
x.shape
y.shape


# 1 訓練データと評価データの準備 ------------------------------------------------------------------

# データ分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# データ確認
# --- Train:Test = 7:3
x_train.shape
y_train.shape
x_test.shape
y_test.shape


# 2 アルゴリズムの選択 ---------------------------------------------------------------------------

# インスタンス作成
# --- 決定木
clf = tree.DecisionTreeClassifier()

# 確認
vars(clf)


# 3 学習 ---------------------------------------------------------------------------------------

# 学習
clf.fit(x_train, y_train)

# 確認
vars(clf)


# 4 予測 ---------------------------------------------------------------------------------------

# 予測
predict = clf.predict(X=x_test)

# モデル評価
# --- 乱数シードを固定していないので書籍と結果が異なる
accuracy_score(y_true=y_test, y_pred=predict)
