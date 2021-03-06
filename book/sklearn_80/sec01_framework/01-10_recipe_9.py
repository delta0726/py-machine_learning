# ******************************************************************************
# Chapter   : 1 機械学習の枠組みを理解する
# Title     : 1-10 機械学習のオーバービュー(Recipe9)
# Created by: Owner
# Created on: 2020/12/20
# Page      : P32 - P39
# ******************************************************************************

# ＜概要＞
# - 分類と回帰の類似点を確認する


# ＜目次＞
# 0 準備
# 1 分類モードの学習
# 2 回帰モードの学習
# 3 モデル精度の評価
# 4 線形と非線形


# 0 準備 -------------------------------------------------------------------------------------------

# ライブラリ
import sklearn
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer


# ライブラリ構成
dir(sklearn.model_selection)
dir(sklearn.svm)
dir(sklearn.metrics)


# データロード
iris = datasets.load_iris()

# 系列作成
# --- ラベル0と1のみを使用（'setosa' / 'versicolor'）
# --- 100レコード
x = iris.data[iris.target < 2]
y = iris.target[iris.target < 2]

# データ分割
# --- 層別サンプリング
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=7)


# 1 分類モードの学習 -----------------------------------------------------------------------------

# 学習器の生成
# --- 分類器
# --- インスタンス生成と学習を同時に実行
svc_clf = SVC(kernel='linear').fit(x_train, y_train)

# クロスバリデーション
svc_score = cross_val_score(svc_clf, x_train, y_train, cv=4)
svc_score.mean()


# 2 回帰モードの学習 -----------------------------------------------------------------------------

# 学習器の生成
# --- 回帰器
svr_clf = SVR(kernel='linear').fit(x_train, y_train)


# 3 モデル精度の評価 -----------------------------------------------------------------------------

# 関数定義
# --- スコア関数
def for_scorer(y_truth, orig_y_pred):
    # 予測値を最も近い整数に丸める
    y_pred = np.rint(orig_y_pred).astype(np.int)
    return accuracy_score(y_truth, y_pred)


# スコアの大小をコントロール
# --- for_scorer()をラップ
svr_to_class_scorer = make_scorer(for_scorer, greater_is_better=True)

# クロスバリデーション
svr_scores = cross_val_score(svc_clf, x_train, y_train, cv=4,
                             scoring=svr_to_class_scorer)

# 平均値の算出
svr_scores.mean()


# 4 線形と非線形 ------------------------------------------------------------------------------

# 学習器の作成
# --- 多項式カーネル
svc_poly_clf = SVC(kernel="poly", degree=3).fit(x_train, y_train)

svc_poly_scores = cross_val_score(svc_poly_clf, x_train, y_train, cv=4)
svc_poly_scores.mean()
