# ******************************************************************************
# Chapter   : 5 ロジスティック回帰
# Title     : 5-3 pandasを使ってデータセットを可視化する（Recipe34)
# Created by: Owner
# Created on: 2020/12/26
# Page      : P134 - P137
# ******************************************************************************

# ＜概要＞
# - データセットの基本的な確認を行う


# ＜目次＞
# 0 準備
# 1 データ確認
# 2 データ可視化


# 0 準備 ------------------------------------------------------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt


# パスの取得
current_path = os.getcwd()
file = os.path.sep.join(['', 'csv', 'pima-indians-diabetes.csv'])

# 列名指定
column_names = ['pregnancy_x',
                'plasma_con',
                'blood_pressure',
                'skin_mm',
                'insulin',
                'bmi',
                'pedigree_func',
                'age',
                'target']

# データ取得
all_data = pd.read_csv(current_path + file,  names=column_names)


# 1 データ確認 --------------------------------------------------------------------------------

# 先頭行の表示
all_data.head()

# 基本統計量の確認
# --- 表示が一部省略されるのでwithブロックで表示変更
with pd.option_context('display.max_columns', 100):
    print(all_data.describe())


# ラベルのカウント
# --- 離散値
all_data.target.value_counts()


# 2 データ可視化 ------------------------------------------------------------------------------

# ヒストグラムの作成
# --- ｢pregnancy_x｣
all_data.pregnancy_x.hist(bins=50)
plt.show()

# ヒストグラムの作成
# --- 全系列
all_data.hist(figsize=(15, 9), bins=50)
plt.show()
