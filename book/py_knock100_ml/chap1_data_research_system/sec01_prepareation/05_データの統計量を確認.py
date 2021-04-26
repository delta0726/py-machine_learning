# ***************************************************************************************
# Title     : Python実践機械学習システム 100本ノック`
# Chapter   : 第1部 データ分析システム
# Section   : 第1章 分析に向けた準備を行う10本ノック
# Theme     : ノック5：データの統計量を確認しよう
# Created by: Owner
# Created on: 2021/4/27
# Page      : P25 - P30
# ***************************************************************************************


# ＜概要＞
# -


# ＜目次＞
# 0 準備
# 1 データ取得
# 2 データの統計量


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
the_order_file = os.path.join(current_dir, "tbl_order_*.csv")
the_order_files = glob.glob(the_order_file)


# 1 データ取得 --------------------------------------------------------------------------

# データフレーム準備
order_all = pd.DataFrame()

# データロード
for file in the_order_files:
    order_data = pd.read_csv(file)
    order_all = pd.concat([order_all, order_data], ignore_index=True)

# データ確認
order_all


# 2 データの統計量 ----------------------------------------------------------------------

# 欠損値の確認
# --- 欠損値なし
order_all.isnull().sum()

# 基本統計量の算出
# --- DataFrame
order_all.describe()

# 基本統計量の算出
# --- Series
order_all['total_amount'].describe()

# 日付の最大値/最小値
order_all['order_accept_date'].min()
order_all['order_accept_date'].max()
order_all['delivered_date'].min()
order_all['delivered_date'].max()
