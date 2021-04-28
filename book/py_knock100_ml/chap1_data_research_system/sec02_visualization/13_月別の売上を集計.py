# ***************************************************************************************
# Title     : Python実践機械学習システム 100本ノック`
# Chapter   : 第1部 データ分析システム
# Section   : 第2章 データを可視化し分析を行う10本ノック
# Theme     : ノック13：月別の売上を集計してみよう
# Created by: Owner
# Created on: 2021/4/28
# Page      : P47 - P50
# ***************************************************************************************


# ＜概要＞
# - データフレームをグループ化して集計する


# ＜目次＞
# 0 準備
# 1 集計列の準備
# 2 売上の月次集計


# 0 準備 -------------------------------------------------------------------------------

# ライブラリ
import os
import pandas as pd


# データロード
os.chdir("book/py_knock100_ml")
order_data = pd.read_csv("data/output_data/order_data.csv")

# データ整形
order_data = order_data.loc[(order_data['status'] == 1) | (order_data['status'] == 2)]
order_data[['store_id', 'coupon_cd']] = order_data[['store_id', 'coupon_cd']].astype(str)
order_data = order_data[['store_id', 'customer_id', 'coupon_cd', 'order_accept_date',
                         'delivered_date', 'total_amount', 'store_name', 'wide_area',
                         'narrow_area', 'takeout_name', 'status_name']]

# データ確認
order_data
order_data.dtypes


# 1 集計列の準備 ----------------------------------------------------------------------

# 集計列の作成
# --- 文字列を日付型に変更
# --- 年月の列を作成
order_data['order_accept_date'] = pd.to_datetime(order_data['order_accept_date'])
order_data['order_accept_month'] = order_data['order_accept_date'].dt.strftime('%Y%m')

# 列のデータ型変更
# --- この章では使用しない
order_data['delivered_date'] = pd.to_datetime(order_data['delivered_date'])
order_data['delivered_month'] = order_data['order_accept_date'].dt.strftime('%Y%m')

# データ確認
order_data[['order_accept_date', 'order_accept_month']].head()
order_data.dtypes

# データ参照
# --- 表記を書籍に合わせる
analyze_data = order_data


# 2 売上の月次集計 ----------------------------------------------------------------------

# ＜ポイント＞
# - グループ化した後の集計は数値列のみで展開される

# グループ化
month_data = analyze_data.groupby('order_accept_month')

# 月次集計
# --- 基本統計量
month_data.describe()

# 月次集計
# --- 合計
month_data.sum()
