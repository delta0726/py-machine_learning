# ***************************************************************************************
# Title     : Python実践機械学習システム 100本ノック`
# Chapter   : 第1部 データ分析システム
# Section   : 第2章 データを可視化し分析を行う10本ノック
# Theme     : ノック16：都道府県別の売上を集計して可視化しよう
# Created by: Owner
# Created on: 2021/4/29
# Page      : P53 - P55
# ***************************************************************************************


# ＜概要＞
# - ピボットテーブルで県ごとの系列を作成してプロットを作成
# - 日本語表記のプロットには{japanize_matplotlib}を使用する


# ＜目次＞
# 0 準備
# 1 データ整形
# 2 都道府県で集計


# 0 準備 -------------------------------------------------------------------------------

# ライブラリ
import os
import pandas as pd

import matplotlib.pyplot as plt
import japanize_matplotlib


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


# 2 都道府県で集計 -----------------------------------------------------------------

# データ確認
analyze_data.loc[:, ['order_accept_month', 'narrow_area', 'total_amount']]

# ピボットテーブル作成
pre_data = pd.pivot_table(analyze_data, index='order_accept_month',
                          columns='narrow_area', values='total_amount',
                          aggfunc='mean')

# プロット作成
plt.plot(list(pre_data.index), pre_data['東京'], label='東京')
plt.plot(list(pre_data.index), pre_data['神奈川'], label='神奈川')
plt.plot(list(pre_data.index), pre_data['埼玉'], label='埼玉')
plt.plot(list(pre_data.index), pre_data['千葉'], label='千葉')
plt.plot(list(pre_data.index), pre_data['茨城'], label='茨城')
plt.plot(list(pre_data.index), pre_data['栃木'], label='栃木')
plt.plot(list(pre_data.index), pre_data['群馬'], label='群馬')
plt.legend()
plt.show()
