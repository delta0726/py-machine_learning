# ******************************************************************************
# Title     : Pythonで学ぶ実験計画法
# Chapter   : 3 データ解析や回帰分析の手法
# Theme     : データの読込
# Date      : 2021/11/07
# Page      : P26 - P30
# ******************************************************************************


# ＜目次＞
# 0 準備
# 1 基本統計量の計算
# 2 共分散と相関係数の計算


# 0 準備 ----------------------------------------------------------------

# ライブラリ
import pandas as pd

# データ準備
df = pd.read_csv('csv/resin.csv', index_col=0, header=0)


# 1 基本統計量の計算 ------------------------------------------------------

# 統計量の結合
# --- 各統計量はPandas Seriesで出力される
statistics = pd.concat([df.mean(), df.median(), df.var(), df.std(),
                        df.max(), df.min(), df.sum()], axis=1).T

# インデックス作成
statistics.index = ['mean', 'median', 'variance', 'standard deviation',
                    'max', 'min', 'sum']

# データ確認
print(statistics)


# ＜参考＞
# メソッドチェーンで記述
# --- インデックス設定はset_axis()で行う
statistics = pd.concat([df.mean(), df.median(), df.var(), df.std(),
                        df.max(), df.min(), df.sum()], axis=1)\
        .T \
        .set_axis(['mean', 'median', 'variance', 'standard deviation',
                   'max', 'min', 'sum'], axis=0)


# 2 共分散と相関係数の計算 ------------------------------------------------

# 分散共分散行列
covariance = df.cov()

# 相関係数行列
correlation_coefficient = df.corr()

# 確認
print(covariance)
print(correlation_coefficient)
