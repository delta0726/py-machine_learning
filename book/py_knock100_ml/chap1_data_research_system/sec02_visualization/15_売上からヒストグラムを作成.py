# ***************************************************************************************
# Title     : Python実践機械学習システム 100本ノック`
# Chapter   : 第1部 データ分析システム
# Section   : 第2章 データを可視化し分析を行う10本ノック
# Theme     : ノック15：売上からヒストグラムを作成してみよう
# Created by: Owner
# Created on: 2021/4/29
# Page      : P52 - P53
# ***************************************************************************************


# ＜概要＞
# - ヒストグラムを作成する


# ＜目次＞
# 0 準備
# 1 データ整形
# 2 ヒストグラム作成


# 0 準備 -------------------------------------------------------------------------------

# ライブラリ
import os
import pandas as pd

import matplotlib.pyplot as plt


# データロード
os.chdir("book/py_knock100_ml")
order_data = pd.read_csv("data/output_data/order_data.csv")


# 1 データ整形 ----------------------------------------------------------------------

# データ整形
order_data = order_data.loc[(order_data['status'] == 1) | (order_data['status'] == 2)]
order_data[['store_id', 'coupon_cd']] = order_data[['store_id', 'coupon_cd']].astype(str)
order_data = order_data[['store_id', 'customer_id', 'coupon_cd', 'order_accept_date',
                         'delivered_date', 'total_amount', 'store_name', 'wide_area',
                         'narrow_area', 'takeout_name', 'status_name']]

# 集計列の作成
# --- 文字列を日付型に変更
# --- 年月の列を作成
order_data['order_accept_date'] = pd.to_datetime(order_data['order_accept_date'])
order_data['order_accept_month'] = order_data['order_accept_date'].dt.strftime('%Y%m')

# データ参照
# --- 表記を書籍に合わせる
analyze_data = order_data


# 2 ヒストグラム作成 -----------------------------------------------------------------

# ヒストグラム作成
# --- ビンの指定なし
plt.hist(analyze_data['total_amount'])
plt.show()

# ヒストグラム作成
# --- ビンの指定あり
plt.hist(analyze_data['total_amount'], bins=21)
plt.show()
