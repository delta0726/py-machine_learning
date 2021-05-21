# ******************************************************************************
# Title     : Scikit-Learnデータ分析実装ハンドブック
# Chapter   : 7 モデル評価
# Theme     : 7-7 分類評価方法
# Created by: Owner
# Created on: 2021/5/22
# Page      : P290 - P292
# ******************************************************************************


# ＜概要＞
# - バイナリ分類問題における評価指標を確認する


# ＜目次＞
# 0 準備
# 1 混合行列
# 2 評価指標


# 0 準備 ------------------------------------------------------------------------------

# ライブラリ
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# 1 混合行列 ---------------------------------------------------------------------------

# データ作成
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 1, 0, 0, 0, 0, 0, 1, 1, 1]

# 混合行列
confusion_matrix(y_true=y_true, y_pred=y_pred)


# 2 評価指標 ----------------------------------------------------------------------------

# 評価指標
# --- 正解率
# --- 適合率
# --- 再現率
# --- F1スコア
accuracy_score(y_true=y_true, y_pred=y_pred)
precision_score(y_true=y_true, y_pred=y_pred)
recall_score(y_true=y_true, y_pred=y_pred)
f1_score(y_true=y_true, y_pred=y_pred)
