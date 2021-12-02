# ***************************************************************************************
# Title     : Pythonによる因果分析（因果推論・因果探索の実践入門）
# Chapter   : 第1章 相関と因果の違いを理解しよう
# Theme     : 3 疑似相関の確認
# Created on: 2021/12/02
# Page      : P25 - P35
# ***************************************************************************************


# ＜概要＞
# - 相関と因果の違いを疑似相関を通して学ぶ


# ＜目次＞
# 0 準備
# 1 ZからYへ因果が存在する場合
# 2 疑似相関：因果が逆
# 3 疑似相関：共通の原因（交絡）
# 4 疑似相関：合流地点での選抜


# 0 準備 -------------------------------------------------------------------------------

# ライブラリ
import random
import numpy as np
import scipy.stats
from numpy.random import randn
import matplotlib.pyplot as plt

# 乱数シードを固定
random.seed(1234)
np.random.seed(1234)


# 1 ZからYへ因果が存在する場合 ------------------------------------------------

# ＜ポイント＞
# - データ生成の際にYにZを含めることでZ⇒Yの因果を表現


# ノイズの生成
# --- 標準正規分布から乱数を生成
num_data = 200
e_z = randn(num_data)
e_y = randn(num_data)

# データの生成
# --- YはZを含んでいる（Z⇒Yの因果関係が存在）
Z = e_z
Y = 2 * Z + e_y

# 相関係数
# --- 出力値は相関があること示すが因果も内包している
np.corrcoef(Z, Y)

# 標準化
Z_std = scipy.stats.zscore(Z)
Y_std = scipy.stats.zscore(Y)

# 散布図をプロット
plt.scatter(Z_std, Y_std)
plt.show()


# 2 疑似相関：因果が逆 -------------------------------------------------------

# ＜ポイント＞
# - 本来はZ⇒Yの因果を想定したデータであるが、Y⇒Zの因果を持つように変更
# - 相関係数も散布図も同じようなものとなり見分けがつかない（疑似相関）


# ノイズの生成
# --- 標準正規分布から乱数を生成
num_data = 200
e_z = randn(num_data)
e_y = randn(num_data)

# データの生成
# --- YはZを含んでいる（Y⇒Zの因果関係が存在）
Y = e_y
Z = 2 * Y + e_z

# 相関係数
# --- 因果が逆であっても相関係数が大きいことがある（疑似相関）
np.corrcoef(Z, Y)

# 標準化
Z_std = scipy.stats.zscore(Z)
Y_std = scipy.stats.zscore(Y)

# 散布図をプロット
plt.scatter(Z_std, Y_std)
plt.show()


# 3 疑似相関：共通の原因（交絡） ----------------------------------------------

# ＜ポイント＞
# - Z⇒Yの因果はないが、YとZが交絡因子Xを持つように変更
# - 交絡因子から間接的な因果関係が生まれて相関を持つようになる（疑似相関）

# ノイズの生成
num_data = 200
e_x = randn(num_data)
e_y = randn(num_data)
e_z = randn(num_data)

# データの生成
Z = 3.3 * e_x + e_z
Y = 3.3 * e_x + e_y

# 相関係数を求める
# --- 因果は無くても相関係数は大きくなる（疑似相関）
np.corrcoef(Z, Y)

# 標準化
Z_std = scipy.stats.zscore(Z)
Y_std = scipy.stats.zscore(Y)

# 散布図をプロット
plt.scatter(Z_std, Y_std)
plt.show()


# 4 疑似相関：合流地点での選抜 -----------------------------------------------

# ＜ポイント＞
# - 選抜前のデータは低相関だったが、選抜後は相関を持つようになる
#   --- 選抜とは一定以上の水準を抽出すること（疑似相関を生み出す）
#   --- 因果関係がなくても相関が生み出される


# 4-1 選抜前のデータを生成 ------------------------------------

# ノイズの生成
num_data = 600
e_x = randn(num_data)
e_y = randn(num_data)

# データの生成
# --- ノイズをそのままデータとする
x = e_x
y = e_y

# 散布図をプロット
# --- 乱数同士の相関なので低相関
plt.scatter(x, y)
plt.show()

# 相関係数
np.corrcoef(x, y)


# 4-2 合流地点での選抜 --------------------------------------

# 合流点を作成
z = x + y

# 新たな合流点での条件を満たす変数の用意
x_new = np.array([])
y_new = np.array([])

# レコード選抜
# --- zの値が0以上のIDのみを選抜
for i in range(num_data):
    if z[i] > 0.0:
        x_new = np.append(x_new, x[i])
        y_new = np.append(y_new, y[i])

# 散布図を描画
plt.scatter(x_new, y_new)
plt.show()

# 相関係数を求める
# --- 左下半分が抜けたことで相関が生まれた
np.corrcoef(x_new, y_new)
