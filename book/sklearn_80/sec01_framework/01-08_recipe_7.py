# ******************************************************************************
# Chapter   : 1 機械学習の枠組みを理解する
# Title     : 1.8 交差検証の紹介(Recipe7)
# Created by: Owner
# Created on: 2020/12/20
# Page      : P25 - P30
# ******************************************************************************


# ＜概要＞
# - irisデータセットは150個しかサンプルがないので偶然モデル精度が高まる可能性がある
#   --- Holdoutはデータ選択結果によりモデル精度が不安定になる可能性
#   --- 交差検証を導入してデータセットを最大限活用して予測精度を測定
# - 交差検証は｢予測精度の評価｣が目的であり、｢予測値の出力｣自体を目的としていない


# ＜目次＞
# 0 準備
# 1 データ分割
# 2 学習と予測
# 3 モデル精度の評価
# 4 交差検証の導入
# 5 層別サンプリング


# 0 準備 -------------------------------------------------------------------------------------------

# ライブラリ
import sklearn

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ライブラリ構成
dir(sklearn.model_selection)
dir(sklearn.svm)
dir(sklearn.linear_model)
dir(sklearn.metrics)


# データ準備
iris = datasets.load_iris()
data = iris.data
target = iris.target

# 系列準備
x = iris.data[:, :2]
y = iris.target


# 1 データ分割 ------------------------------------------------------------------

# ＜ポイント＞
# - データセットが異なることによる予測精度のバラツキを調べるため複数データを作成


# データ分割
# --- データ全体を訓練/テストデータに分割
# --- 25％をテストデータとする
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.25, random_state=7)

# データ分割-2
# --- 訓練データをさらに訓練/テストデータに分割
# --- 25％をテストデータとする
x_train_2, x_test_2, y_train_2, y_test_2 = \
    train_test_split(x_train, y_train, test_size=0.25, random_state=7)

# 確認
len(x_test) / len(data)
len(x_test_2) / len(x_train)
len(x_test_2) / len(data)


# 2 学習と予測 ------------------------------------------------------------------------

# ＜ポイント＞
# - 学習器の生成は｢x_train_2｣データを使用する


# 学習器の作成
# --- サポートベクターマシン
# --- ロジスティック回帰
clf_svc = SVC(kernel='linear', random_state=7)
clf_lr = LogisticRegression(random_state=7)

# 学習
clf_svc.fit(x_train_2, y_train_2)
clf_lr.fit(x_train_2, y_train_2)

# 予測2
# --- ｢データ分割-2｣のデータを使用
# --- 学習器はx_train_2で訓練
pred_svc_2 = clf_svc.predict(x_test_2)
pred_lr_2 = clf_lr.predict(x_test_2)

# 予測1
# --- ｢データ分割-1｣のデータを使用
# --- 学習器はx_train_2で訓練
pred_svc_1 = clf_svc.predict(x_test)
pred_lr_1 = clf_lr.predict(x_test)


# 3 モデル精度の評価 ------------------------------------------------------------------------

# ＜ポイント＞
# - データセットによってAccuracyが大きく変化することを確認


# モデル精度の取得
# --- 2回目の分割
print('Accuracy of SVC:', accuracy_score(y_test_2, pred_svc_2))
print('Accuracy of LR:', accuracy_score(y_test_2, pred_lr_2))

# モデル精度の取得
# --- 1回目の分割
print('Accuracy of SVC:', accuracy_score(y_test, pred_svc_1))
print('Accuracy of LR:', accuracy_score(y_test, pred_lr_1))


# 4 交差検証の導入 -------------------------------------------------------------------------

# クロスバリデーションの実行
# --- 訓練データ全体を用いる(x_train)
# --- 学習器(estimator)は未学習の状態でよい
scores_svc = cross_val_score(estimator=clf_svc, X=x_train, y=y_train, cv=4)
scores_lr = cross_val_score(estimator=clf_lr, X=x_train, y=y_train, cv=4)


# スコアの平均と標準偏差
# --- SVC
print('SVC_AVG:', scores_svc.mean())
print('SVC_STD:', scores_svc.std())

# スコアの平均と標準偏差
# --- LR
print('SVC_AVG:', scores_lr.mean())
print('SVC_STD:', scores_lr.std())


# 5 層別サンプリング ----------------------------------------------------------------------

# ＜ポイント＞
# - モデルの偏りを減らすため層化サンプリングを行う
#   --- 分類問題においてラベル(y)に対して層化サンプリングを行うのはセオリー


# データ分割
# --- 層化サンプリング
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y)

# クロスバリデーション
scores_svc = cross_val_score(clf_svc, x_train, y_train, cv=4)
print("SVC_AVG:", scores_svc.mean())
print("SVC_STD:", scores_svc.std())

# Accuracy
print("Accuracy on Final Test Set:", accuracy_score(y_test, clf_svc.predict(x_test)))
