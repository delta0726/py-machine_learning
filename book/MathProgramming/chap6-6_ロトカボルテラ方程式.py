# ******************************************************************************
# Title       : AI・データサイエンスのための数学プログラミング
# Chapter     : 6 感染症の影響を予測してみよう
# Theme       : 6-6 生物間や競合他社の競争を説明するロトカボルテラ方程式
# Creat Date  : 2021/1/13
# Final Update:
# Page        : P269 - P272
# ******************************************************************************


# ＜概要＞
# - ねずみ算で増加する一方で生物には生存競争(住処/食料など)があるため、単純なねずみ算は成立しない
#   --- この考え方は企業の製品競争などにも応用できる


# ＜目次＞
# 0 準備
# 1 ロトカボルテラ方程式
# 2 アニメーション


# 0 準備 ---------------------------------------------------------------

# ライブラリ
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc


# 1 ロトカボルテラ方程式 ---------------------------------------------------

# ＜ポイント＞
# - ロトカボルテラ方程式は時間発展方程式のcapacityを複数種の合計により設定する
#   --- キャパシティに対して競争が発生する


# 共通パラメータ
# --- 期間変化の単位
# --- シミュレーション回数
dt = 1.0
num = 10

# 個別パラメータ（生物種1）
# --- 初期値
# --- キャパシティ
# --- 微分係数（増加ペース）
r1 = 1
K1 = 110
a = 0.1

# 個別パラメータ（生物種2）
# --- 初期値
# --- キャパシティ
# --- 微分係数（増加ペース）
r2 = 1
K2 = 80
b = 1.1

# 初期化（初期値設定）
n1 = np.zeros(num)
n2 = np.zeros(num)
n1[0] = 2
n2[0] = 2

# 時間発展方程式
# --- capacityを生物種1と生物種2の合計で決まるように設定している
for t in range(1, num):
    delta_n1 = int(r1 * n1[t - 1] * (1 - (n1[t - 1] + a * n2[t - 1]) / K1))
    n1[t] = delta_n1 * dt + n1[t - 1]
    delta_n2 = int(r2 * n2[t - 1] * (1 - (n2[t - 1] + b * n1[t - 1]) / K2))
    n2[t] = delta_n2 * dt + n2[t - 1]

plt.plot(n1, label='n1')
plt.plot(n2, label='n2')
plt.legend()
plt.show()


# 2 アニメーション ----------------------------------------------------------------

# パラメータ設定
dt = 1.0
r1 = 1
K1 = 110
a = 0.1
r2 = 1
K2 = 80
b = 1.1
num = 10
x_size = 8.0
y_size = 6.0

# 初期化（初期値設定）
n1 = np.zeros(num)
n2 = np.zeros(num)
n1[0] = 2
n2[0] = 2
list_plot = []

# 時間発展方程式
fig = plt.figure()
for t in range(1, num):
    delta_n1 = int(r1 * n1[t - 1] * (1 - (n1[t - 1] + a * n2[t - 1]) / K1))
    n1[t] = delta_n1 * dt + n1[t - 1]
    delta_n2 = int(r2 * n2[t - 1] * (1 - (n2[t - 1] + b * n1[t - 1]) / K2))
    n2[t] = delta_n2 * dt + n2[t - 1]
    x_n1 = np.random.rand(int(n1[t])) * x_size
    y_n1 = np.random.rand(int(n1[t])) * y_size
    img = [plt.scatter(x_n1, y_n1, color="blue")]
    x_n2 = np.random.rand(int(n2[t])) * x_size
    y_n2 = np.random.rand(int(n2[t])) * y_size
    img += [plt.scatter(x_n2, y_n2, color="red")]
    list_plot.append(img)

# グラフ（アニメーション）描画
plt.grid()
anim = animation.ArtistAnimation(fig, list_plot, interval=200, repeat_delay=1000)
rc('animation', html='jshtml')
plt.close()
anim
