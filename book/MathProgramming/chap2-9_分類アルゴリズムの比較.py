# ******************************************************************************
# Title       : AI・データサイエンスのための数学プログラミング
# Chapter     : 2 機械学習を使った分析を行ってみよう
# Theme       : 2-9 様々な分類アルゴリズムの比較
# Creat Date  : 2021/12/23
# Final Update:
# Page        : P86 - P88
# ******************************************************************************


# ＜概要＞
# - Scikit-Learnはモデル構文が統一されているのでモデル切替えが容易に行える


# ＜目次＞
# 0 準備
# 1 ランダムフォレストによる分類
# 2 SVMによる分類


# 0 準備 ---------------------------------------------------------------------

# ライブラリ
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# データ読み込み
df_info = pd.read_csv("csv/accomodation_data.csv", index_col=0)

# データ作成
features = df_info.loc[:, lambda x: x.columns.str.startswith("X")]
label = df_info.loc[:, lambda x: x.columns.str.startswith("Y")]

# データ分割
x_train, x_test, y_train, y_test = train_test_split(features, label)


# 1 ランダムフォレストによる分類 ------------------------------------------------

# ＜ポイント＞
# - ランダムフォレストは特徴量とインスタンスをランダムに抽出して決定木を作成してアンサンブルする
#   --- 決定木がもつ不安定性を解消


# モデル構築＆学習
model = RandomForestClassifier(bootstrap=True, n_estimators=10, max_depth=None, random_state=1)
clf = model.fit(x_train, y_train)

# スコア計算
score = clf.score(x_test, y_test)
print("スコア:", score)

# 混同行列の作成
pred_tree = clf.predict(x_test)
cm = confusion_matrix(y_test, pred_tree)
print("混同行列:")
print(cm)


# 2 SVMによる分類 --------------------------------------------------------------

# ＜ポイント＞
# - SVMは空間の中で空間敵に分割を行っていく


# モデル構築＆学習
model = SVC(kernel='rbf')
clf = model.fit(x_train, y_train)

# スコア計算
score = clf.score(x_test, y_test)
print("スコア:", score)

# 混同行列の作成
pred_tree = clf.predict(x_test)
cm = confusion_matrix(y_test, pred_tree)
print("混同行列:")
print(cm)
