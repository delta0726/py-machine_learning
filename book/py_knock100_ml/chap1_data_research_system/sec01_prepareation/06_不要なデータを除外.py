# ***************************************************************************************
# Title     : Python実践機械学習システム 100本ノック`
# Chapter   : 第1部 データ分析システム
# Section   : 第1章 分析に向けた準備を行う10本ノック
# Theme     : ノック6：不要なデータを除外しよう
# Created by: Owner
# Created on: 2021/4/27
# Page      : P30 - P31
# ***************************************************************************************


# ＜概要＞
# - 不要データをフィルタで削除する


# ＜目次＞
# 0 準備
# 1 データ取得
# 2 不要データの削除


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


# 2 不要データの削除 ---------------------------------------------------------------------

# 系列確認
# --- 999が除外対象データ
order_all.store_id.value_counts().sort_index().to_dict()

# データのフィルタ
# --- 999を除外
order_all = order_all.loc[order_all['store_id'] != 999]
