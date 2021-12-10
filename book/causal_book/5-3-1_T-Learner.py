# ***************************************************************************************
# Title     : Pythonによる因果分析（因果推論・因果探索の実践入門）
# Chapter   : 第5章 機械学習を用いた因果推論
# Theme     : 3-1 T-Learner
# Created on: 2021/12/07
# Page      : P108 - P110
# ***************************************************************************************


# ＜概要＞
# - T-Learnerとは介入の有無でデータセットを2つに分けて2つのモデルを構築してアプローチする
#   --- モデルを2つ作成することで反実仮想を表現する


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
# - ATEは平均処置効果のことで集団レベルでの因果効果を指す
# - ATEは作成した2つのモデルの予測値の差分の平均値として定義される
#   --- 予測には全レコードを用いている点に注意


# 予測値の算出
# --- データセットは全レコードを用いている点に注意
pred_0 = reg_0.predict(df[["x"]])
pred_1 = reg_1.predict(df[["x"]])

# ATEの算出
ATE = (pred_1 - pred_0).mean()
print("ATE：", ATE)


# 4 ATTとATUの算出 ---------------------------------------------------------------

# ＜ポイント＞
# - ATTとは処置群における平均処置効果、ATEとは対象群における平均処置効果
# - それぞれの処置効果を求めるのに必要な反実仮想の結果を回帰モデルから計算
#   --- 作成したモデルと異なる方のデータセットで予測することで反実仮想を表現


# 予測値の算出
pred_0_df_1 = reg_0.predict(df_1[["x"]])
pred_1_df_0 = reg_1.predict(df_0[["x"]])

# 処置群における平均処置効果（ATT）
# --- 正解と予測値の差分
ATT = (df_1["Y"] - pred_0_df_1).mean()

# 対照群における平均処置効果（ATU）
# --- 予測値と正解の差分
ATU = (pred_1_df_0 - df_0["Y"]).mean()

# 確認
print("ATT：", ATT)
print("ATU：", ATU)


# 5 プロット作成 -----------------------------------------------------------------

# ＜ポイント＞
# - 各人ごとの推定された治療効果がダミーデータの仮定を再現できているかを確認


# データ作成
# --- 各人ごとの推定された治療効果（ATEとして平均する前のデータ）
t_estimated = pred_1 - pred_0

# データ作成
# --- 正解データのイメージ（面談の満足度の閾値を示すステップデータ）
# --- 参考: P105-107
x_index = np.arange(-1, 1, 0.01)
t_ans = np.zeros(len(x_index))
for i in range(len(x_index)):
    if x_index[i] < 0:
        t_ans[i] = 0.5
    elif 0 <= x_index[i] < 0.5:
        t_ans[i] = 0.7
    elif x_index[i] >= 0.5:
        t_ans[i] = 1.0

# プロット作成
# --- 後の方法と比べると当てはまりはさほどよくない
plt.scatter(df[["x"]], t_estimated, label="estimated_treatment-effect")
plt.plot(x_index, t_ans, color='black', ls='--', label='Baseline')
plt.ylim(0.4, 1.1)
plt.show()
