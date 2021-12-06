# ***************************************************************************************
# Title     : Pythonによる因果分析（因果推論・因果探索の実践入門）
# Chapter   : 第5章 機械学習を用いた因果推論
# Theme     : 3-1 T-Learner
# Created on: 2021/12/07
# Page      : P108 - P110
# ***************************************************************************************


# ＜概要＞
# - T-Learnerとは介入の有無でデータセットを2つに分けて2つのモデルを構築してアプローチする


# ＜目次＞
# 0 準備
# 1 データ作成
# 2 モデル構築
# 3 ATEの算出
# 4 ATTとATUの算出
# 5 プロット作成


# 0 準備 ---------------------------------------------------------------------------

# ライブラリ
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


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


# 2 モデル構築 --------------------------------------------------------------------

# ＜ポイント＞
# - 介入効果の有無で分割した各データセットでモデル構築を行う


# 介入を受けていないモデル
reg_0 = RandomForestRegressor(max_depth=3)
reg_0.fit(df_0[["x"]], df_0[["Y"]])

# 介入を受けたモデル
reg_1 = RandomForestRegressor(max_depth=3)
reg_1.fit(df_1[["x"]], df_1[["Y"]])


# 3 ATEの算出 --------------------------------------------------------------------

# ＜ポイント＞
# - ATEは作成した2つのモデルの予測値の差分の平均値として定義される


# 予測値の算出
# --- モデルは異なるが、データセットは全レコードを用いている点に注意
mu_0 = reg_0.predict(df[["x"]])
mu_1 = reg_1.predict(df[["x"]])

# ATEの算出
ATE = (mu_1 - mu_0).mean()
print("ATE：", ATE)


# 4 ATTとATUの算出 ---------------------------------------------------------------

# ＜ポイント＞
# - それぞれの処置効果を求めるのに必要な反実仮想の結果を回帰モデルから計算


# 処置群における平均処置効果（ATT）
# --- 正解と予測値の差分
ATT = df_1["Y"] - reg_0.predict(df_1[["x"]])

# 対照群における平均処置効果（ATU）
# --- 予測値と正解の差分
ATU = reg_1.predict(df_0[["x"]]) - df_0["Y"]

# 確認
print("ATT：", ATT.mean())
print("ATU：", ATU.mean())


# 5 プロット作成 -----------------------------------------------------------------

# 推定された治療効果を各人ごとに求めます
t_estimated = reg_1.predict(
    df[["x"]]) - reg_0.predict(df[["x"]])
plt.scatter(df[["x"]], t_estimated,
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
plt.show()
