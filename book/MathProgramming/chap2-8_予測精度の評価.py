# ******************************************************************************
# Title       : AI・データサイエンスのための数学プログラミング
# Chapter     : 2 機械学習を使った分析を行ってみよう
# Theme       : 2-8 予測精度の評価の流れを理解しよう
# Creat Date  : 2021/12/22
# Final Update:
# Page        : P84 - P85
# ******************************************************************************


# ＜概要＞
# - 機械学習では訓練データでモデルを作り、検証データで評価する


# ＜目次＞
# 0 準備
# 1 モデル構築
# 2 予測精度の評価


# 0 準備 ---------------------------------------------------------------------

# ライブラリ
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


# データ読み込み
df_info = pd.read_csv("csv/accomodation_data.csv", index_col=0)

# データ作成
features = df_info.loc[:, lambda x: x.columns.str.startswith("X")]
label = df_info.loc[:, lambda x: x.columns.str.startswith("Y")]


# 1 モデル構築 ------------------------------------------------------------------

# ＜ポイント＞
# - データを訓練データと検証データに分割して、訓練データでモデルを構築する


# データ分割
x_train, x_test, y_train, y_test = train_test_split(features, label)

# モデル構築
clf = DecisionTreeClassifier(max_depth=2)
clf = clf.fit(x_train, y_train)

# 確認
vars(clf)


# 2 予測精度の評価 --------------------------------------------------------------

# ＜ポイント＞
# - 訓練データで構築したモデルを検証データで評価する
# - 分類問題の場合はメトリックに加えて混合行列も見るほうが実感が得やすい


# スコア計算
score = clf.score(x_test, y_test)
print("スコア:", score)

# 混同行列の作成
pred_tree = clf.predict(x_test)
cm = confusion_matrix(y_test, pred_tree)
print("混同行列")
print(cm)
