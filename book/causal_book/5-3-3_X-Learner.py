# ***************************************************************************************
# Title     : Pythonによる因果分析（因果推論・因果探索の実践入門）
# Chapter   : 第5章 機械学習を用いた因果推論
# Theme     : 3-3 X-Learner
# Created on: 2021/12/08
# Page      : P112 - P115
# ***************************************************************************************


# ＜概要＞
# - X-Learnerとは傾向スコアを用いてT-Learnerの結果を更に補正するアプローチ


# ＜目次＞
# 0 準備
# 1 モデル構築
# 2 ATEの算出
# 3 プロット作成


# 0 準備 ---------------------------------------------------------------------------

# ライブラリ
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression


# データロード
df = pd.read_csv('csv/career.csv')

# データ確認
print(df)


# 1 データ作成 ---------------------------------------------------------------------

# ＜ポイント＞
# - 介入効果を検証したい項目でデータセットを分割する


# 集団を2つに分ける
# --- 介入を受けていない集団（研修を未受講）
# --- 介入を受けた集団（研修を受講）
df_0 = df[df.Z == 0.0]
df_1 = df[df.Z == 1.0]

# レコード数の確認
df_0.shape
df_1.shape


# 2 T-Learnerによる効果推定 --------------------------------------------------------

# ＜ポイント＞
# - 介入効果の有無で分割した各データセットでモデル構築を行う


# 介入を受けていないモデル
M0 = RandomForestRegressor(max_depth=3)
M0.fit(df_0[["x"]], df_0[["Y"]])

# 介入を受けたモデル
M1 = RandomForestRegressor(max_depth=3)
M1.fit(df_1[["x"]], df_1[["Y"]])

# 予測値の算出
pred_0 = M1.predict(df_0[["x"]])
pred_1 = M0.predict(df_1[["x"]])

# 治療効果の推定
# --- 推定された治療効果を各人ごとに求めます
tau_0 = pred_0 - df_0["Y"]
tau_1 = df_1["Y"] - pred_1


# 3 ATTとATUの算出 -------------------------------------------------------------

# ＜ポイント＞
# - 治療効果の推定値をYとしてモデルを構築する


# ATTを求めるモデル
# --- 処置群における平均処置効果
M2 = RandomForestRegressor(max_depth=3)
M2.fit(df_0[["x"]], tau_0)

# ATUを求めるモデル
# --- 対照群における平均処置効果
M3 = RandomForestRegressor(max_depth=3)
M3.fit(df_1[["x"]], tau_1)

# ATTとATUの算出
ATT = M2.predict(df[["x"]])
ATU = M2.predict(df[["x"]])


# 4 傾向スコアによる補正 -------------------------------------------------------------

# ＜ポイント＞
# - 全体のデータを用いてロジスティック回帰モデルを定義して傾向スコアを算出する
# - ATTとATUを傾向スコアで調整する


# 変数定義
# --- 説明変数
# --- 被説明変数（目的変数）
X = df[["x"]]
Z = df["Z"]

# モデル構築
g_x = LogisticRegression().fit(X, Z)

# クラス確率の予測
# --- 傾向スコア
score = g_x.predict_proba(X)

# 効果を傾向スコアで調整
# --- ATT： 処置群
# --- ATU： 対照群
tau = ATT * score[:, 1] + ATU * score[:, 0]


# 5 プロット作成 ---------------------------------------------------------------------

# ＜ポイント＞
# - X-Learnerの方がS-Learnerよりも各人の研修効果の推定値が正解の点線に収束している


# 散布図の作成
# --- 推定された治療効果を各人ごとに求めます
plt.scatter(df[["x"]], tau, label="estimated_treatment-effect")

# 正解のグラフを作成
x_index = np.arange(-1, 1, 0.01)
t_ans = np.zeros(len(x_index))
for i in range(len(x_index)):
    if x_index[i] < 0:
        t_ans[i] = 0.5
    elif 0 <= x_index[i] < 0.5:
        t_ans[i] = 0.7
    elif x_index[i] >= 0.5:
        t_ans[i] = 1.0

# 正解を描画
plt.plot(x_index, t_ans, color='black', ls='--', label='Baseline')
plt.ylim(0.4, 1.1)

# プロット表示
plt.show()
