# ***************************************************************************************
# Title     : 意思決定分析と予測の活用
# Chapter   : 第2部 決定分析の基本
# Theme     : 第2章 Pythonの導入
# Created on: 2021/4/24
# Page      : P57 - P64
# ***************************************************************************************


# ＜概要＞
# - Pythonの基本操作の確認


# ＜目次＞
# 0 準備
# 1 リスト
# 2 関数の作成
# 3 numpyのndarray
# 4 pandasのデータフレーム
# 5 pandasのシリーズ
# 6 データフレームの演算


# 0 準備 -------------------------------------------------------------------------------

# ライブラリ
import numpy as np
import pandas as pd


# 1 リスト -------------------------------------------------------------------------------

# リストの作成
nums = [1, 2, 3]
nums


# 2 関数の作成 ---------------------------------------------------------------------------

# 関数定義
def my_func(in_value):
    out_value = in_value * 10
    return(out_value)

# 関数実行
my_func(in_value=5)

# 関数実行
# --- 引数省略
my_func(5)

# 3 numpyのndarray ---------------------------------------------------------------------

# 配列定義
array_1 = np.array([1, 2])
array_1

# 配列のメリット
# --- 配列を用いると計算が楽になる
# --- リストではエラーとなる（[1, 2] + 1）
array_1 + 1


# 4 pandasのデータフレーム -------------------------------------------------------------

# データフレーム定義
df_1 = pd.DataFrame({
    'column1': [1, 2, 3],
    'column2': [4, 5, 6],
}, index=['row1', 'row2', 'row3'])

# 確認
df_1


# 5 pandasのシリーズ ------------------------------------------------------------------

# シリーズ定義
series_1 = pd.Series([7, 8], index=['idx1', 'idx2'])
series_1

# データフレームからの列抽出
# --- シリーズとして抽出される
series_2 = df_1['column1']
series_2


# 6 データフレームの演算 -----------------------------------------------------------------

# データ確認
df_1

# 加算
df_1 + 1

# 加算
# --- メソッドを使用
df_1.add(1)

# 列ごとの最大値
df_1.max()

# 行ごとの最大値
df_1.max(axis=1)

# 関数適用
df_1.apply(np.log2)

# 列ごとの合計
df_1.apply(np.sum, axis=0)

# 行ごとの合計
df_1.apply(np.sum, axis=1)
df_1.sum(axis=1)
