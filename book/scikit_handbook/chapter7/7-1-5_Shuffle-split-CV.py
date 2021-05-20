# ******************************************************************************
# Title     : Scikit-Learnデータ分析実装ハンドブック
# Chapter   : 7 モデル評価
# Theme     : 7-5 シャッフル交差検証
# Created by: Owner
# Created on: 2021/5/21
# Page      : P288 - P289
# ******************************************************************************


# ＜概要＞
# - 大規模データセットだと時間がかかりすぎるので非現実的
#   --- ごく小規模のデータセットにしか使えない


# ＜目次＞
# 0 準備
# 1 シャッフル交差検証


# 0 準備 ------------------------------------------------------------------------------

# ライブラリ
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score


# データロード
iris = load_iris()

# データ格納
X = iris.data
y = iris.target


# 1 シャッフル交差検証 ------------------------------------------------------------------

# インスタンス生成
# --- モデル
model = LogisticRegression()

# インスタンス生成
# --- ランダムにnセットの訓練データ/テストデータの抽出が行われる
# --- 必ずしも訓練データ+テストデータが1となる必要はない
# --- ランダム抽出なので、n_splitとは別にtest_sizeを設定することができる
shuffle_split = ShuffleSplit(n_splits=10, random_state=0, test_size=0.5, train_size=0.5)

# モデル評価
# --- shuffle_splitをcv引数に設定
scores = cross_val_score(model, X, y, cv=shuffle_split)

# データ確認
# --- データセットのレコード数と同値
len(scores)
scores