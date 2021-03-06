# ******************************************************************************
# Chapter   : 1 機械学習の枠組みを理解する
# Title     : 1.7 最も小さな機械学習レシピ(Recipe6)
# Created by: Owner
# Created on: 2020/12/20
# Page      : P21 - P25
# ******************************************************************************


# ＜概要＞
# - 機械学習は予測を行うことを目的としている
# - 以下のワークフローに基づいてアプローチする


# ＜ワークフロー＞
# - 1 解決すべき問題の明文化
# - 2 モデル選択
# - 3 モデル訓練
# - 4 予測
# - 5 モデルの性能評価


# ＜目次＞
# 0 準備
# 1 データ準備
# 2 学習と予測(SVM)
# 3 学習と予測(ロジスティック回帰)


# 0 準備 -------------------------------------------------------------------------------------------

# ライブラリ
import sklearn

from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# sklearnの構成確認
dir(sklearn)
dir(sklearn.svm)
dir(sklearn.model_selection)
dir(sklearn.linear_model)
dir(sklearn.metrics)


# データ準備
# --- 特徴量とラベルを分離しておく
# --- 機械学習コミュニティではこのように定義するのが慣例
iris = datasets.load_iris()
data = iris.data
target = iris.target


# 1 データ準備 *******************************************************

# ＜ポイント＞
# - 教師あり分類問題として扱う


# 系列準備
x = iris.data[:, :2]
y = iris.target

# データ分割
# --- 25％を検証データとする
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.25, random_state=1)

# 確認
len(x_test) / len(x)


# 2 学習と予測(SVM) *******************************************************

# ＜ポイント＞
# - サポートベクターマシン(SVM)の分類器を用いて学習する


# インスタンス作成
# --- パラメータを初期値から変更する引数のみ設定
# --- 引数一覧はCtrl+Pでヘルプ表示して確認
clf_svm = SVC(kernel='linear', random_state=1)
clf_svm

# 学習
clf_svm.fit(x_train, y_train)

# 予測
y_pred = clf_svm.predict(x_test)

# 予測精度の測定
# --- 76.3％
accuracy_score(y_test, y_pred)


# 3 学習と予測(ロジスティック回帰) ********************************************

# ＜ポイント＞
# - ロジスティック回帰の分類器を用いて学習する
#   --- SVMのベンチマークとして別のアルゴリズムで計算


# インスタンスの作成
clf_log = LogisticRegression(random_state=1)
clf_log

# 学習
clf_log.fit(x_train, y_train)

# 予測
y_pred = clf_log.predict(x_test)

# 予測精度の測定
# --- 60.5％
accuracy_score(y_test, y_pred)
