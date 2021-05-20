# ******************************************************************************
# Title     : Scikit-Learnデータ分析実装ハンドブック
# Chapter   : 7 モデル評価
# Theme     : 7-1 ホールドアウト法
# Created by: Owner
# Created on: 2021/5/20
# Page      : P285
# ******************************************************************************


# ＜概要＞
# - データセット全体を訓練データとテストデータに分割する
#   --- 訓練データ：モデル構築に使用
#   --- テストデータ：汎化性能評価に使用


# ＜関数＞
# - train_test_split()
#   --- データ分割を行う関数であり、モデル評価自体は行わない


# ＜目次＞
# 0 準備
# 1 データ分割
# 2 モデル精度の評価


# 0 準備 ------------------------------------------------------------------------------

# ライブラリ
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# データロード
iris = load_iris()

# データ格納
X = iris.data
y = iris.target


# 1 データ分割 ----------------------------------------------------------------------------

# データ分割
# --- ホールドアウト法
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# データ確認
X_train.shape
X_test.shape
y_train.shape
y_test.shape


# 2 モデル精度の評価 ----------------------------------------------------------------------------

# モデル構築
model = LogisticRegression()

# モデル学習
model.fit(X_train, y_train)

# 予測
accuracy_score(y_true=y_test, y_pred=model.predict(X_test))
