# ***************************************************************************************
# Title     : Pythonによる因果分析（因果推論・因果探索の実践入門）
# Chapter   : 第5章 機械学習をもちいた因果推論
# Theme     : 1 ランダムフォレストによる分類と回帰の仕組み
# Created on: 2021/12/04
# Page      : P94 - P103
# ***************************************************************************************


# ＜概要＞
# - 決定木とランダムフォレストを分類/回帰のそれぞれで扱う
# - 情報利得の定義が分類と回帰で異なる点に注意


# ＜目次＞
# 0 準備
# 1 決定木での分類
# 2 決定木での回帰
# 3 ランダムフォレストでの分類
# 4 ランダムフォレストでの回帰


# 0 準備 ------------------------------------------------------------------------

# ライブラリ
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

# データロード
df = pd.read_csv('csv/tv_cm.csv')

# データ確認
print(df)


# 1 決定木での分類 -----------------------------------------------------------------

# ＜ポイント＞
# - 分類の場合は情報利得にジニ不純度かエントロピーが用いられる（デフォルトはジニ不純度）


# 変数定義
# --- 説明変数
# --- 目的変数
X = df[["年齢", "性別"]]
Z = df["CMを見た"]

# データ分割
# --- 訓練データと検証データ
X_train, X_val, Z_train, Z_val = train_test_split(
    X, Z, train_size=0.6, random_state=0)

# モデル構築＆学習
# --- max_depth=1
clf = DecisionTreeClassifier(max_depth=1, random_state=0)
clf.fit(X_train, Z_train)
print("深さ1の性能：", clf.score(X_val, Z_val))

# モデル構築＆学習
# --- max_depth=2
clf = DecisionTreeClassifier(max_depth=2, random_state=0)
clf.fit(X_train, Z_train)
print("深さ2の性能：", clf.score(X_val, Z_val))

# モデル構築＆学習
# --- max_depth=3
clf = DecisionTreeClassifier(max_depth=3, random_state=0)
clf.fit(X_train, Z_train)
print("深さ3の性能：", clf.score(X_val, Z_val))


# 2 決定木での回帰 -----------------------------------------------------------------

# ＜ポイント＞
# - 回帰の場合は情報利得にジニ係数ではなく二乗誤差が使われる


# 変数定義
# --- 説明変数
# --- 目的変数
X = df[["年齢", "性別"]]
Y = df["購入量"]

# データ分割
# --- 訓練データと検証データ
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, train_size=0.6, random_state=0)

# モデル構築＆学習
# --- max_depth=2
reg = DecisionTreeRegressor(max_depth=2, random_state=0)
reg = reg.fit(X_train, Y_train)
print("深さ2の性能：", reg.score(X_val, Y_val))

# モデル構築＆学習
# --- max_depth=3
reg = DecisionTreeRegressor(max_depth=3, random_state=0)
reg = reg.fit(X_train, Y_train)
print("深さ3の性能：", reg.score(X_val, Y_val))

# モデル構築＆学習
# --- max_depth=4
reg = DecisionTreeRegressor(max_depth=4, random_state=0)
reg = reg.fit(X_train, Y_train)
print("深さ4の性能：", reg.score(X_val, Y_val))


# 3 ランダムフォレストでの分類 ----------------------------------------------------

# ＜ポイント＞
# - ランダムフォレストは特徴量と学習データをランダムに抽出して大領のツリーを作成してアンサンブルする
# - 分類の場合は情報利得にジニ不純度かエントロピーが用いられる（デフォルトはジニ不純度）


# 変数定義
# --- 説明変数
# --- 目的変数
X = df[["年齢", "性別"]]
Z = df["CMを見た"]

# データ分割
# --- 訓練データと検証データ
X_train, X_val, Z_train, Z_val = train_test_split(
    X, Z, train_size=0.6, random_state=0)

# モデル構築＆学習
# --- max_depth=1
clf = RandomForestClassifier(max_depth=1, random_state=0)
clf.fit(X_train, Z_train)
print("深さ1の性能：", clf.score(X_val, Z_val))

# モデル構築＆学習
# --- max_depth=2
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, Z_train)
print("深さ2の性能：", clf.score(X_val, Z_val))

# モデル構築＆学習
# --- max_depth=3
clf = RandomForestClassifier(max_depth=3, random_state=0)
clf.fit(X_train, Z_train)
print("深さ3の性能：", clf.score(X_val, Z_val))


# 4 ランダムフォレストでの回帰 ----------------------------------------------------

# 変数定義
# --- 説明変数
# --- 目的変数
X = df[["年齢", "性別"]]
Y = df["購入量"]

# データ分割
# --- 訓練データと検証データ
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, train_size=0.6, random_state=0)

# モデル構築＆学習
# --- max_depth=2
reg = RandomForestRegressor(max_depth=2, random_state=0)
reg = reg.fit(X_train, Y_train)
print("深さ2の性能：", reg.score(X_val, Y_val))

# モデル構築＆学習
# --- max_depth=3
reg = RandomForestRegressor(max_depth=3, random_state=0)
reg = reg.fit(X_train, Y_train)
print("深さ3の性能：", reg.score(X_val, Y_val))

# モデル構築＆学習
# --- max_depth=4
# --- 決定係数R2を表示
reg = RandomForestRegressor(max_depth=4, random_state=0)
reg = reg.fit(X_train, Y_train)
print("深さ4の性能：", reg.score(X_val, Y_val))
vars(reg)
