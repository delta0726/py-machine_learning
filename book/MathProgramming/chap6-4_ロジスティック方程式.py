# ******************************************************************************
# Title       : AI・データサイエンスのための数学プログラミング
# Chapter     : 6 感染症の影響を予測してみよう
# Theme       : 6-4 実際の生物や社会の現象を説明するロジスティック方程式
# Creat Date  : 2021/1/12
# Final Update:
# Page        : P263 - P266
# ******************************************************************************


# ＜概要＞
# - ロジスティック方程式とは時間発展方程式にキャパシティの概念を導入したもの


# ＜目次＞
# 0 準備
# 1 時間発展方程式にキャパシティを導入
# 2 アニメーションによる確認


# 0 準備 ---------------------------------------------------------------------

# ライブラリ
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc


# 1 時間発展方程式にキャパシティを導入 -------------------------------------------

# ＜ポイント＞
# - deltaにキャパシティの項を追加する
#   --- プロットがキャパシティで


# パラメータ設定
# --- 期間変化量（固定値）
# --- 微分係数（増加ペース）
# --- キャパシティ
# --- シミュレーション回数
dt = 1.0
a = 1.2
capacity = 100
num = 20

# 初期化（初期値設定）
n = np.zeros(num)
n[0] = 2

# 時間発展方程式
# --- deltaに(1 - n[t - 1] / capacity))の項を追加
# --- キャパシティ直前まで通常どおり増加
# --- キャパシティに到達したら項が0となるため値を維持
for t in range(1, num):
    delta = int(a * n[t - 1] * (1 - n[t - 1] / capacity))
    n[t] = delta * dt + n[t - 1]

# プロット作成
plt.plot(n)
plt.show()


# 2 アニメーションによる確認 -------------------------------------------

# パラメータ設定
dt = 1.0
a = 1.2
capacity = 100
num = 20
x_size = 8.0
y_size = 6.0

# 初期化（初期値設定）
n = np.zeros(num)
n[0] = 2
list_plot = []

# 時間発展方程式
fig = plt.figure()
for t in range(1, num):
    delta = int(a * n[t - 1] * (1 - n[t - 1] / capacity))
    n[t] = delta * dt + n[t - 1]
    x_n = np.random.rand(int(n[t])) * x_size
    y_n = np.random.rand(int(n[t])) * y_size
    img = plt.scatter(x_n, y_n, color="black")
    list_plot.append([img])

# グラフ（アニメーション）描画
plt.grid()
anim = animation.ArtistAnimation(fig, list_plot, interval=200, repeat_delay=1000)
rc('animation', html='jshtml')
plt.close()
anim
