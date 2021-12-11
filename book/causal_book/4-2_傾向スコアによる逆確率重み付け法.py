# ***************************************************************************************
# Title     : Pythonによる因果分析（因果推論・因果探索の実践入門）
# Chapter   : 第4章 因果推論を実装しよう
# Theme     : 2 傾向スコアを用いた逆確率重み付け法(IPTW)の実装
# Created on: 2021/12/04
# Page      : P81 - P86
# ***************************************************************************************


# ＜概要＞
# - 処置群と統制群の分類問題から傾向スコアを定義する（ラベルの問題設定がポイント）
# - 傾向スコアの逆数で加重することで群ごとの期待効果を求めて差分効果を算出する


# ＜傾向スコアの意味＞
# - 傾向スコアとはY(介入有無)をX(変数)で分類予測する際のクラス確率
#   --- Yが1/0それぞれの場合におけるXの傾向度合いを示す(尤もらしさ)
#   --- 特徴量におけるサンプリングバイアスの度合いを示す


# ＜参考＞
# 傾向スコアとIPW推定量の基本的な考え方
# https://www.trifields.jp/basic-idea-of-propensity-score-and-ipw-estimator-by-statistical-causal-inference-2650


# ＜目次＞
# 0 準備
# 1 モデル構築
# 2 傾向スコアの算出
# 3 平均処置効果(ATE)の算出


# 0 準備 ------------------------------------------------------------------------

# ライブラリ
import pandas as pd
from sklearn.linear_model import LogisticRegression


# データロード
df = pd.read_csv('csv/tv_cm.csv')

# データ確認
print(df)


# 1 モデル構築 ------------------------------------------------------------------

# ＜ポイント＞
# - 傾向スコアは分類問題のクラス確率に基づいて定義される
# - アルゴリズムは何でもよいが、ロジスティック回帰を使用するのが一般的
# - ｢傾向スコア｣は処置を受ける確率を意味するので処置を受ける場合を｢1｣としてラベル付けする
#   --- 1を処置群、0を統制群という


# 変数定義
# --- 説明変数
# --- 目的変数
X = df[["年齢", "性別"]]
Z = df["CMを見た"]

# モデル構築
# --- 介入有無をYとして分類問題モデルを構築
reg = LogisticRegression().fit(X, Z)

# データ確認
# --- 回帰した結果の係数を出力
# --- 仮想データ生成の際に設定したパラメータがほぼ再現されている
# --- 0.1 * (x_1 + (1 - x_2) * 10 - 40 + 5 * e_z)
print("係数beta：", reg.coef_)
print("係数alpha：", reg.intercept_)


# 2 傾向スコアの算出 --------------------------------------------------------------

# ＜ポイント＞
# - 処置群を｢1｣としているので、傾向スコアは｢処置をする確率｣｢処置しない確率｣の順で表示される


# クラス確率の算出
# --- 傾向スコア
Z_pre = reg.predict_proba(X)

# データ確認
# --- 予測確率と正解ラベルの表示
pd.concat([pd.DataFrame(Z_pre[0:5]),
           pd.DataFrame(Z[0:5])], axis=1)


# 3 平均処置効果(ATE)の算出 --------------------------------------------------------

# ＜ポイント＞
# - 傾向スコアの逆数でウエイトを付けて効果を算出する（傾向スコアが1だった場合の期待購入金額を推定）
# - 群ごとに期待購入金額を集計して差分で効果を求める
# - サンプル数で割ることで平均処置効果を算出する

# 購入金額
Y = df['購入量']

# サンプルごとの期待購入金額
# --- 購入金額を傾向スコアで調整（スコアが1だった場合の金額を推定）
# --- CMを見たかどうかでZは1/0を出力（第1項と第2項のどちらかのみ出力）
ATE_i = Y * 1 / Z_pre[:, 1] * Z - Y * 1 / Z_pre[:, 0] * (1 - Z)

# 平均処置効果
# --- 1サンプルあたりの平均効果
ATE = ATE_i.sum() / len(Y)

# データ確認
print("推定したATE", ATE)
