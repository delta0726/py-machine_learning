# ***************************************************************************************
# Title     : Python実践機械学習システム 100本ノック`
# Chapter   : 第1部 データ分析システム
# Section   : 第2章 データを可視化し分析を行う10本ノック
# Theme     : ノック11：データを読み込んで不要なものを除外しよう
# Created by: Owner
# Created on: 2021/4/28
# Page      : P42 - P44
# ***************************************************************************************


# ＜概要＞
# - 1章で加工したデータを読込んで、分析に必要な形に整形する


# ＜目次＞
# 0 準備
# 1 データ整形


# 0 準備 -------------------------------------------------------------------------------

import os
import pandas as pd


# カレントディレクトリ設定
os.chdir("book/py_knock100_ml")

# データロード
order_data = pd.read_csv("data/output_data/order_data.csv")

# データ確認
order_data.shape
order_data.info()
order_data.head()


# 1 データ整形 ---------------------------------------------------------------------------

# 不要データの削除
# --- status：0（受付）
# --- status：9（キャンセル）
order_data = order_data.loc[(order_data['status'] == 1) | (order_data['status'] == 2)]
print(order_data)
order_data.columns

# 列の選択
analyze_data = order_data[['store_id', 'customer_id', 'coupon_cd', 'sales_detail_id',
                           'order_accept_date', 'delivered_date', 'total_amount',
                           'store_name', 'wide_area', 'narrow_area', 'takeout_name',
                           'status_name']]
print(analyze_data)
analyze_data.columns
