# ******************************************************************************
# Title     : Pythonで学ぶ実験計画法
# Chapter   : 3 データ解析や回帰分析の手法
# Theme     : ヒストグラムと散布図
# Date      : 2021/11/07
# Page      : P24 - P26
# ******************************************************************************


# ＜目次＞
# 0 準備
# 1 ヒストグラムの作成
# 2 散布図の作成


# 0 準備 ----------------------------------------------------------------

# ライブラリ
import pandas as pd
import matplotlib.pyplot as plt


# データ準備
df = pd.read_csv('csv/resin.csv', index_col=0, header=0)


# 1 ヒストグラムの作成 ----------------------------------------------------

# パラメータ設定
# --- ヒストグラムを描画する特徴量の番号
# --- ビンの数
number_of_variable = 0
number_of_bins = 10

# ヒストグラム作成
plt.rcParams['font.size'] = 12
plt.hist(df.iloc[:, number_of_variable], bins=number_of_bins)
plt.xlabel(df.columns[number_of_variable])
plt.ylabel('frequency')
plt.show()


# 2 散布図の作成 --------------------------------------------------------

# パラメータ設定
# --- 散布図における横軸の特徴量の番号
# --- 散布図における縦軸の特徴量の番号
variable_number_1 = 0
variable_number_2 = 1

# 散布図の作成
plt.rcParams['font.size'] = 12
plt.scatter(df.iloc[:, variable_number_1], 
            df.iloc[:, variable_number_2])
plt.xlabel(df.columns[variable_number_1])
plt.ylabel(df.columns[variable_number_2])
plt.show()
