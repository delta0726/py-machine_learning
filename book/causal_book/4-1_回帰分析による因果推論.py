# ***************************************************************************************
# Title     : Pythonによる因果分析（因果推論・因果探索の実践入門）
# Chapter   : 第4章 因果推論を実装しよう
# Theme     : 1 回帰分析による因果推論の実装
# Created on: 2021/12/03
# Page      : P74 - P80
# ***************************************************************************************


# ＜概要＞
# - 因果の大きさを推定する方法として線形回帰分析を学ぶ
#   --- 目的変数から説明変数の共通効果を取り除く操作（従来の回帰分析と着目点が異なる）


# ＜目次＞
# 0 準備
# 1 データ確認
# 2 線形回帰モデルによる確認


# 0 準備 ------------------------------------------------------------------------

# ライブラリ
import pandas as pd
from sklearn.linear_model import LinearRegression

# データロード
df = pd.read_csv('csv/tv_cm.csv')


# 1 データ確認 -------------------------------------------------------------------

# ＜ポイント＞
# - 効果検証では施策の有無による差分に着目する


# 平均値の確認
# --- CMを見た人： 高齢 / 女性 / 購入量：少
# --- CMを見てない人： 若年 / 男性 / 購入量：大
df[df["CMを見た"] == 1.0].mean()
df[df["CMを見た"] == 0.0].mean()


# 2 線形回帰モデルによる確認 ------------------------------------------------------

# ＜ポイント＞
# - 今回使用するデータは入力変数に対してd分離が行われて残った変数のみを使用している
# - 変数を回帰モデルに説明変数として投入することは、変数のバックドアパスを閉じることにつながる
#   --- 目的変数から説明変数の効果を除外する意味を持つ


# 変数定義
# --- 説明変数
# --- 目的変数
X = df[["年齢", "性別", "CMを見た"]]
y = df["購入量"]

# モデル構築
reg = LinearRegression().fit(X, y)

# 回帰係数の確認
# --- 仮想データの想定したパラメータをほぼ再現している
print("係数：", reg.coef_)
