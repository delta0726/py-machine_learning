# ***************************************************************************************
# Title     : Pythonによる因果分析（因果推論・因果探索の実践入門）
# Chapter   : 第5章 機械学習を用いた因果推論
# Theme     : 2 Meta-Learnerの実装
# Created on: 2021/12/06
# Page      : P104 - P107
# ***************************************************************************************


# ＜概要＞
# - 乱数を用いてMeta-Learner用の仮想データを作成する
#   --- ｢上司向け：部下とのキャリア面談のポイント研修｣データ


# ＜目次＞
# 0 準備
# 1 データの作成
# 2 データフレーム作成
# 3 プロット確認


# 0 準備 ---------------------------------------------------------------------------

# ライブラリ
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import randn
from scipy.special import expit

# 乱数のシードを設定
np.random.seed(1234)
random.seed(1234)


# 1 データの作成 -------------------------------------------------------------------

# ＜ポイント＞
# - ｢上司が研修を受講したか(Z)｣が介入効果となっている
#   --- Zの効果の係数(t)は上司の熱心さ(x)にステップ関数で連動するものとして定義


# データ数
num_data = 500


# 部下育成への熱心さ(x)
# --- -1から1の一様乱数
x = np.random.uniform(low=-1, high=1, size=num_data)


# 上司が研修を受講したか(Z)
# --- ノイズの生成（標準正規乱数）
# --- シグモイド変換
e_z = randn(num_data)
z_prob = expit(-1 * -5.0 * x + 5 * e_z)

# 受講フラグの生成
# 上司が「上司向け：部下とのキャリア面談のポイント研修」に参加したかどうか
i = 1
Z = np.array([])
for i in range(num_data):
    Z_i = np.random.choice(2, size=1, p=[1 - z_prob[i], z_prob[i]])[0]
    Z = np.append(Z, Z_i)


# 部下の面談の満足度(Y)
# --- 介入効果の定義
# --- 部下育成の熱心さ[x]の値に応じて段階的に変化
t = np.zeros(num_data)
for i in range(num_data):
    if x[i] < 0:
        t[i] = 0.5
    elif 0 <= x[i] < 0.5:
        t[i] = 0.7
    elif x[i] >= 0.5:
        t[i] = 1.0

# 最終的な満足度
# --- Y = Z * t(x) + 0.3 * x + 2 + 0.1 * Noise
e_y = randn(num_data)
Y = Z * t + 0.3 * x + 2.0 + 0.1 * e_y


# 2 データフレーム作成 -----------------------------------------------------

# データフレーム作成
df = pd.DataFrame({'x': x,
                   'Z': Z,
                   't': t,
                   'Y': Y,
                   })

# データ確認
df.head()

# データ保存
# df.to_csv('csv/career.csv')


# 3 プロット確認 ----------------------------------------------------------

# 介入効果を図で確認
# --- x   : 部下の育成熱心さ
# --- t(x): 研修効果の大きさ
plt.scatter(x, t, label="treatment-effect")
plt.show()

# 散布図
# --- x: 部下の育成熱心さ
# --- Y: 部下の面談の満足度
plt.scatter(x, Y)
plt.show()
