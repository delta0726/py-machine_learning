# ***************************************************************************************
# Title     : Pythonによる因果分析（因果推論・因果探索の実践入門）
# Chapter   : 第5章 機械学習を用いた因果推論
# Theme     : 4 DR-Learner
# Created on: 2021/12/09
# Page      : P116 - P120
# ***************************************************************************************


# ＜概要＞
# - DR法は回帰分析とIPTW法を組み合わせた方法
# - DR-Learnerとは反実仮想(潜在的結果変数)の推定に利用して、より性能の高い機械学習モデルを構築する手法


# ＜目次＞
# 0 準備
# 1 データ作成
# 2 T-Learnerによる効果推定
# 3 傾向スコアの算出
# 4 ITEの算出
# 5 プロット作成


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


# 3 傾向スコアの算出 -------------------------------------------------------------

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

# 傾向スコアの算出
# --- クラス確率の予測
score = g_x.predict_proba(X)


# 4 ITEの算出 ------------------------------------------------------------------------

# 変数定義（処置群）
X1 = df_1[["x"]]
Y1 = df_1["Y"]

# 変数定義（対照群）
X0 = df_0[["x"]]
Y0 = df_0["Y"]

# 処置群
# 非処置群
# [:,1]はZ=1側の確率
Y_1 = M1.predict(X1) + (Y1 - M1.predict(X1)) / g_x.predict_proba(X1)[:, 1]
Y_0 = M0.predict(X0) + (Y0 - M0.predict(X0)) / g_x.predict_proba(X0)[:, 0]

# [:,0]はZ=0側の確率
df_1["ITE"] = Y_1 - M0.predict(X1)
df_0["ITE"] = M1.predict(X0) - Y_0

# 表を結合する
df_DR = pd.concat([df_0, df_1])
df_DR.head()


# 5 プロット作成 ---------------------------------------------------------------------

# ＜ポイント＞
# - X-Learnerの方がS-Learnerよりも各人の研修効果の推定値が正解の点線に収束している


# モデルM_DR
M_DR = RandomForestRegressor(max_depth=3)
M_DR.fit(df_DR[["x"]], df_DR[["ITE"]])


# 推定された治療効果を各人ごとに求めます
t_estimated = M_DR.predict(df_DR[["x"]])
plt.scatter(df_DR[["x"]], t_estimated,
            label="estimated_treatment-effect")

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

# プロット表示
plt.show()
