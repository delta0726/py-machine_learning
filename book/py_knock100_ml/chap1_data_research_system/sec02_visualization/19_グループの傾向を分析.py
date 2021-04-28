# ***************************************************************************************
# Title     : Python実践機械学習システム 100本ノック`
# Chapter   : 第1部 データ分析システム
# Section   : 第2章 データを可視化し分析を行う10本ノック
# Theme     : ノック19：グループの傾向を分析してみよう
# Created by: Owner
# Created on: 2021/4/29
# Page      : P60 - P61
# ***************************************************************************************


# ＜概要＞
# - クラスタリングで作成したグループごとに集計を行う


# ＜目次＞
# 0 準備
# 1 データ整形
# 2 クラスタリング


# 0 準備 -------------------------------------------------------------------------------

# ライブラリ
import os
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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

# データ集計
# --- ストアごとの統計量
store_clustering = analyze_data\
    .groupby('store_id')\
    .agg(['size', 'mean', 'median', 'max', 'min'])['total_amount']\
    .astype(float)\
    .reset_index(drop=True)

# データ確認
store_clustering


# 2 クラスタリング ----------------------------------------------------------

# データ基準化
# --- 前処理
sc = StandardScaler()
store_clustering_sc = sc.fit_transform(store_clustering)

# クラスタリング実行
kmeans = KMeans(n_clusters=4, random_state=0)
clusters = kmeans.fit(store_clustering_sc)

# データセット加工
# --- クラスタリング結果を追加
store_clustering['clusters'] = clusters.labels_
store_clustering.columns = ['月内件数', '月内平均値', '月内中央値',
                            '月内最大値', '月内最小値', 'cluster']

# クラスタ集計
# --- 書籍と結果が異なるが（クラスタの割り当てイメージは一致）
# --- ランダムシードを固定しているので、何故一致しないかは不明
store_clustering.groupby('cluster').count()

# クラスタ平均
store_clustering.groupby('cluster').mean()
