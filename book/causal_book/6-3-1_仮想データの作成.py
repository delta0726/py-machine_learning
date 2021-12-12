# ***************************************************************************************
# Title     : Pythonによる因果分析（因果推論・因果探索の実践入門）
# Chapter   : 第6章 LINGAMの実装
# Theme     : LiNGAMによる因果探索の実装（データ作成）
# Created on: 2021/12/13
# Page      : P131 - P132
# ***************************************************************************************


# ＜概要＞
# - 因果探索に使用するデータを作成する


# ＜目次＞
# 0 準備
# 1 データ作成


# 0 準備 ---------------------------------------------------------------------------

# ライブラリ
import random
import numpy as np
import pandas as pd

# 乱数シードの固定
random.seed(1234)
np.random.seed(1234)


# 1 データ作成 ----------------------------------------------------------------------

# データ数
num_data = 200

# 非ガウスのノイズ
# --- -1.0から1.0の一様分布
ex1 = 2 * (np.random.rand(num_data) - 0.5)
ex2 = 2 * (np.random.rand(num_data) - 0.5)
ex3 = 2 * (np.random.rand(num_data) - 0.5)

# データ生成
x2 = ex2
x1 = 3 * x2 + ex1
x3 = 2 * x1 + 4 * x2 + ex3

# 表にまとめる
df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
df.head()

# データ保存
# df.to_csv('csv/lingam_data.csv')
