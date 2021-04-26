# ***************************************************************************************
# Title     : Python実践機械学習システム 100本ノック`
# Chapter   : 第1部 データ分析システム
# Section   : 第1章 分析に向けた準備を行う10本ノック
# Theme     : ノック7：マスタデータを結合（ジョイン）してみよう
# Created by: Owner
# Created on: 2021/4/27
# Page      : P31 - P33
# ***************************************************************************************


# ＜概要＞
# - テーブル同士を結合する


# ＜目次＞
# 0 準備
# 1 注文データ取得
# 2 その他のデータの取得
# 3 テーブル結合


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

# データ準備
# --- 店舗マスタの取得
# --- 地域マスタの取得
m_store = pd.read_csv("m_store.csv")
m_area = pd.read_csv("m_area.csv")


# 1 注文データ取得 -----------------------------------------------------------------------

# ファイル一覧の取得
the_order_file = os.path.join(current_dir, "tbl_order_*.csv")
the_order_files = glob.glob(the_order_file)

# データフレーム準備
order_all = pd.DataFrame()

# データロード
for file in the_order_files:
    order_data = pd.read_csv(file)
    order_all = pd.concat([order_all, order_data], ignore_index=True)

# 不要データの削除
order_all = order_all.loc[order_all['store_id'] != 999]


# 2テーブル結合 ------------------------------------------------------------------------

# 列の確認
order_data.columns
m_store.columns
m_area.columns

# 結合
# --- store_id
order_data = pd.merge(order_data, m_store, on="store_id", how="left")
order_data

# 結合
# --- area_cd
order_data = pd.merge(order_data, m_area, on="area_cd", how="left")
order_data

# 確認
order_data.columns
