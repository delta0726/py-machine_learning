# ***************************************************************************************
# Title     : Pythonによる因果分析（因果推論・因果探索の実践入門）
# Chapter   : 第4章 因果推論を実装しよう
# Theme     : 3 Doubly Robust法による因果推論のぞssぴ
# Created on: 2021/12/04
# Page      : P87 - P91
# ***************************************************************************************


# ＜概要＞
# - DR法は回帰分析とIPTW法を組み合わせた方法
# - 回帰分析で反実仮想の推定値を用いてIPTW法を行う（1/0の両方の情報を使えるようになる）


# ＜目次＞
# 0 準備
# 1 回帰モデルから施策の効果を測定
# 2 傾向スコアの推定
# 3 平均処置効果(ATE)の算出


# 0 準備 ------------------------------------------------------------------------

# ライブラリ
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


# データロード
df = pd.read_csv('csv/tv_cm.csv')

# データ確認
print(df)


# 1 回帰モデルから施策の効果を測定 ------------------------------------------

# ＜ポイント＞
# - 施策の項(CMを見た)を含めて線形回帰モデルを構築
# - モデルに対して施策の項(CMを見た)を全て1/0に変更した際の購入量を計算
#   --- 反実仮想の購入額の期待値を推定している（実測値ではない点に注意）


# 変数定義
# --- 説明変数
# --- 目的変数
X = df[["年齢", "性別", "CMを見た"]]
y = df["購入量"]

# モデル構築
# --- 線形回帰モデル
reg2 = LinearRegression().fit(X, y)

# 回帰係数の確認
reg2.coef_

# CMを見なかった場合の購入量
# --- Z=0
X_0 = X.copy()
X_0["CMを見た"] = 0
Y_0 = reg2.predict(X_0)

# CMを見た場合の購入量
# --- Z=1
X_1 = X.copy()
X_1["CMを見た"] = 1
Y_1 = reg2.predict(X_1)


# 2 傾向スコアの推定 ---------------------------------------------------------

# ＜ポイント＞
# - 傾向スコアの算出方法は同じ


# 変数定義
# --- 説明変数
# --- 目的変数
X = df[["年齢", "性別"]]
Z = df["CMを見た"]

# モデル構築
# --- 線形分類モデル（ロジスティック回帰）
reg = LogisticRegression().fit(X, Z)

# 傾向スコアの算出
Z_pre = reg.predict_proba(X)


# 3 平均処置効果(ATE)の算出 ---------------------------------------------------

# ＜ポイント＞
# - サンプルごとの期待購入金額を1/0の両方のケースで算出
#   --- 推定値を用いることで反実仮想の制約を超えてATEを算出することができる


# 購入金額
Y = df['購入量']

# サンプルごとの期待購入金額
ATE_1_i = Y / Z_pre[:, 1] * Z + (1 - Z / Z_pre[:, 1]) * Y_1
ATE_0_i = Y / Z_pre[:, 0] * (1 - Z) + (1 - (1 - Z) / Z_pre[:, 0]) * Y_0

# 平均処置効果
# --- 1サンプルあたりの平均効果
ATE = (ATE_1_i - ATE_0_i).sum() / len(Y)

# データ確認
print("推定したATE", ATE)
