# ******************************************************************************
# Title     : Scikit-Learnデータ分析実装ハンドブック
# Chapter   : 7 モデル評価
# Theme     : 7-4 1つ抜き交差検証
# Created by: Owner
# Created on: 2021/5/21
# Page      : P288
# ******************************************************************************


# ＜概要＞
# - 大規模データセットだと時間がかかりすぎるので非現実的
#   --- ごく小規模のデータセットにしか使えない


# ＜目次＞
# 0 準備
# 1 1つ抜き交差検証


# 0 準備 ------------------------------------------------------------------------------

# ライブラリ
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score


# データロード
iris = load_iris()

# データ格納
X = iris.data
y = iris.target


# 1 1つ抜き交差検証 ---------------------------------------------------------------------

# インスタンス生成
# --- モデル
model = LogisticRegression()

# インスタンス生成
# --- コントロール余地がないため引数もない
lo = LeaveOneOut()

# モデル評価
# --- loをcv引数に設定
scores = cross_val_score(model, X, y, cv=lo)

# データ確認
# --- データセットのレコード数と同値
len(scores)
