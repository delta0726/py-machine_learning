# ***************************************************************************************
# Title     : Python実践機械学習システム 100本ノック`
# Chapter   : 第1部 データ分析システム
# Section   : 第1章 分析に向けた準備を行う10本ノック
# Theme     : ノック4：複数データを結合（ユニオン）してみよう
# Created by: Owner
# Created on: 2021/4/25
# Page      : P25 - P27
# ***************************************************************************************


# ＜概要＞
# - 同じ形式のファイルを読込時に結合する


# ＜目次＞
# 0 準備
# 1 1ファイルの読込
# 2 複数ファイルを読込時に結合


# 0 準備 -------------------------------------------------------------------------------

# ライブラリ
import os
import glob
import pandas as pd


# カレントパスの指定
# --- ファイル保存先
os.chdir("book/py_knock100_ml/data")

# カレントパスの取得
current_dir = os.getcwd()

# ファイル一覧の取得
the_order_file = os.path.join(current_dir, 'tbl_order_*.csv')
the_order_files = glob.glob(the_order_file)


# 1 1ファイルの読込 --------------------------------------------------------------------

# データフレーム準備
order_all = pd.DataFrame()

# ファイル取得
file = the_order_files[0]

# データ取得
order_data = pd.read_csv(file)

# データ結合
order_all = pd.concat([order_all, order_data], ignore_index=True)

# 確認
len(order_data)
len(order_all)


# 2 複数ファイルを読込時に結合 ----------------------------------------------------------

# データフレーム準備
order_all = pd.DataFrame()

# 読込時に結合
# --- file = the_order_files[1]
for file in the_order_files:
    order_data = pd.read_csv(file)
    print(f'{file}:{len(order_data)}')
    order_all = pd.concat([order_all, order_data], ignore_index=True)

# データ確認
order_all
