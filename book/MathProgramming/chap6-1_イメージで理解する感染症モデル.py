# ******************************************************************************
# Title       : AI・データサイエンスのための数学プログラミング
# Chapter     : 6 感染症の影響を予測してみよう
# Theme       : 6-1 イメージで理解する感染症モデル
# Creat Date  : 2021/1/10
# Final Update:
# Page        : P250 - P254
# ******************************************************************************


# ＜概要＞
# - 感染症の予測を行うための基本モデルはSIRモデルが知られている（微分方程式）
#   --- 感受性保持者(S)、感染者(I)、免疫保持者(R)よりSIRと表現する
#   --- 微分方程式をドミノ倒しに例えてforループで表現する


# ＜目次＞
# 0 準備
# 1 初期値の設定
# 2 シミュレーション
# 3 プロット作成


# 0 準備 ---------------------------------------------------------------------

# ライブラリ
import numpy as np
import matplotlib.pyplot as plt


# 1 初期値の設定 ---------------------------------------------------------------

# ＜ポイント＞
# - 時系列で100回のシミュレーションを行う
# - 時間発展方程式はパラメータを設定すれば将来の値を決めることができる（forループ）


# シミュレーション回数
# --- 日数
num = 100

# 配列生成
# --- 感受性保持者(Sus)
# --- 感染者(Inf)
# --- 免疫保持者(Rec)
sus = np.zeros(num)
inf = np.zeros(num)
rec = np.zeros(num)

# SIRパラメータ
S = 200000
I = 2
R = 0
alpha = I / (S + I + R)

# 初期値
# --- 感受性保持者(S)
# --- 感染者(I)
# --- 免疫保持者(R)
sus[0] = S
inf[0] = I
rec[0] = R


# 2 シミュレーション --------------------------------------------------

# 微分パラメータ
dt = 1.0
beta = 0.000003

# 感染者が翌期に回復する割合
gamma = 0.1

# 時間発展方程式
t = 1
for t in range(1, num):
    # パラメータ取得
    S = sus[t - 1]
    I = inf[t - 1]
    R = rec[t - 1]

    # alpha = I / (S + I + R)

    # 時刻t-1からtへの変化分の計算
    # --- 翌期回復者数（I ⇒ R）
    # --- 全体の増分（S ⇒ I）
    delta_R = I * gamma
    delta_S = -beta * S * I
    if delta_S > 0:
        delta_S = 0
    delta_I = -delta_S - delta_R

    # 時刻tでの値の計算
    I = I + delta_I * dt
    R = R + delta_R * dt
    S = S + delta_S * dt
    if S < 0:
        S = 0
    sus[t] = S
    inf[t] = I
    rec[t] = R


# 3 プロット作成 ----------------------------------------------------------

# グラフ描画
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(sus, label="S(susceptible)", color="orange")
plt.plot(inf, label="I(infection)", color="blue")
plt.plot(rec, label="R(recover)", color="green")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(inf, label="I(infection)", color="blue")
plt.legend()
plt.show()
