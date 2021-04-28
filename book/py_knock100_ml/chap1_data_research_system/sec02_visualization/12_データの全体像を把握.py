# ***************************************************************************************
# Title     : Python実践機械学習システム 100本ノック`
# Chapter   : 第1部 データ分析システム
# Section   : 第2章 データを可視化し分析を行う10本ノック
# Theme     : ノック12：データの全体像を把握しよう
# Created by: Owner
# Created on: 2021/4/28
# Page      : P45 - P47
# ***************************************************************************************


# ＜概要＞
# -


# ＜目次＞
# 0 準備
# 1 データセットの確認
# 2 警告が出ないようにする


# 0 準備 -------------------------------------------------------------------------------

# ライブラリ
import os
import pandas as pd

import warnings


# データロード
os.chdir("book/py_knock100_ml")
order_data = pd.read_csv("data/output_data/order_data.csv")

# データ整形
order_data = order_data.loc[(order_data['status'] == 1) | (order_data['status'] == 2)]
analyze_data = order_data[['store_id', 'customer_id', 'coupon_cd', 'order_accept_date',
                           'delivered_date', 'total_amount', 'store_name', 'wide_area',
                           'narrow_area', 'takeout_name', 'status_name']]

# データ確認
analyze_data


# 1 データセットの確認 ------------------------------------------------------------------

# 基本統計量
analyze_data.describe()

# データ型の確認
# --- objectは文字列を意味する
analyze_data.dtypes

# IDを文字列に変更
# --- 警告が出る
# --- ｢analyze_dataの参照元であるorder_dataのデータ型を変更すべき｣という警告
analyze_data[['store_id', 'coupon_cd']] = analyze_data[['store_id', 'coupon_cd']].astype(str)

# データ型の確認
analyze_data.dtypes


# 2 警告が出ないようにする ------------------------------------------------------------------

# ＜ポイント＞
# - 予備知識として警告を出さない方法を確認する
#   --- 本来は警告が出ないような書き方をすべき

# 警告メッセージを非表示
warnings.filterwarnings('ignore')

# IDを文字列に変更
# --- 警告が出ない
analyze_data[['store_id', 'coupon_cd']] = analyze_data[['store_id', 'coupon_cd']].astype(str)
