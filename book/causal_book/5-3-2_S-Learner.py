# ***************************************************************************************
# Title     : Pythonによる因果分析（因果推論・因果探索の実践入門）
# Chapter   : 第5章 機械学習を用いた因果推論
# Theme     : 3-2 S-Learner
# Created on: 2021/12/08
# Page      : P110 - P112
# ***************************************************************************************


# ＜概要＞
# - S-Learnerとは1つのモデルに対して介入有無を想定した2つのデータセットを用いてアプローチする


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


# データロード
df = pd.read_csv('csv/career.csv')

# データ確認
print(df)


# 1 モデル構築 --------------------------------------------------------------------

# ＜ポイント＞
# - データの該当列を選択したうえで全レコードを用いてモデルを構築する
#   --- モデルは予測ではなく説明に使うのでデータ分割は必要ない


# データ抽出
X = df.loc[:, ["x", "Z"]]
y = df.loc[:, ["Y"]]

# モデル定義
reg = RandomForestRegressor(max_depth=4)

# モデル学習
# --- 全レコードを使用
reg.fit(X, y)


# 2 ATEの算出 --------------------------------------------------------------------

# ＜ポイント＞
# - モデルに対して介入効果を1/0に統一したデータセットで予測を行う
# - ATEは2つの予測値の差分の平均値として定義される


# 2群のデータ作成
# --- 全てのレコードを処置が0又は1の状態のデータを作成する
X_0 = X.assign(Z=0)
X_1 = X.assign(Z=1)

# 予測値の算出
# --- 2群データとして加工したデータを用いて予測
mu_0 = reg.predict(X_0)
mu_1 = reg.predict(X_1)

# ATEの算出
ATE = (mu_1 - mu_0).mean()
print("ATE：", ATE)


# 3 プロット作成 -----------------------------------------------------------------

# 推定効果の算出
# --- 推定された治療効果を各人ごとに算出
t_estimated = mu_1 - mu_0

# 予測値をプロット
plt.scatter(df[["x"]], t_estimated, label="estimated_treatment-effect")

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
