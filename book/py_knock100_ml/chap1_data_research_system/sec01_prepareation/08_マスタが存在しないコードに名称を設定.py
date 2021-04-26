# ***************************************************************************************
# Title     : Python実践機械学習システム 100本ノック`
# Chapter   : 第1部 データ分析システム
# Section   : 第1章 分析に向けた準備を行う10本ノック
# Theme     : ノック8：マスタが存在しないコードに名称を設定しよう
# Created by: Owner
# Created on: 2021/4/27
# Page      : P33 - P35
# ***************************************************************************************


# ＜概要＞
# - フラグ系列を説明する系列を追加する


# ＜目次＞
# 0 準備
# 1 注文データ取得
# 2 テーブル結合
# 3 系列追加1：takeout_flag
# 4 系列追加2：takeout_flag


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


# 2 テーブル結合 -----------------------------------------------------------------

# 結合
order_data = pd.merge(order_data, m_store, on="store_id", how="left")
order_data = pd.merge(order_data, m_area, on="area_cd", how="left")
order_data


# 3 系列追加1：takeout_flag ----------------------------------------------------

# ＜ポイント＞
# - フラグ1/0を名称に変更する

# 系列要素の確認
order_data['takeout_flag'].value_counts()

# 系列追加
order_data.loc[order_data['takeout_flag'] == 0, 'takeout_name'] = 'デリバリー'
order_data.loc[order_data['takeout_flag'] == 1, 'takeout_name'] = 'お持ち帰り'

# データ確認
order_data


# 4 系列追加2：takeout_flag ----------------------------------------------------

# 系列要素の確認
order_data['status'].value_counts()

# 系列追加
order_data.loc[order_data['status'] == 0, 'status_name'] = '受付'
order_data.loc[order_data['status'] == 1, 'status_name'] = 'お支払い済'
order_data.loc[order_data['status'] == 2, 'status_name'] = 'お渡し済'
order_data.loc[order_data['status'] == 9, 'status_name'] = 'キャンセル'

# データ確認
order_data
