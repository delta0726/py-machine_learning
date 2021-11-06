# ******************************************************************************
# Title     : Pythonで学ぶ実験計画法
# Chapter   : 3 データ解析や回帰分析の手法
# Theme     : 特徴量の標準化
# Date      : 2021/11/07
# Page      : P31 - P32
# ******************************************************************************


# ＜目次＞
# 0 準備
# 1 特徴量の標準化
# 2 メソッドチェーンで記述


# 0 準備 ----------------------------------------------------------------

# ライブラリ
import pandas as pd

# データ準備
df = pd.read_csv('csv/resin.csv', index_col=0, header=0)


# 1 特徴量の標準化 ---------------------------------------------------------

# 特徴量の削除
# --- 標準偏差がゼロの特徴量（ゼロ・バリアンス・フィルタ）
deleting_variables = df.columns[df.std() == 0]
df = df.drop(deleting_variables, axis=1)

# 特徴量の標準化
scaled_df = (df - df.mean()) / df.std()

# 確認
print(scaled_df)
print(scaled_df.mean())
print(scaled_df.std())


# 2 メソッドチェーンで記述 -------------------------------------------------

# 特徴量の標準化
scaled_df = df\
    .drop(df.columns[df.std() == 0], axis=1)\
    .transform(lambda x: (x - x.mean()) / x.std())

# 確認
print(scaled_df)
print(scaled_df.mean())
print(scaled_df.std())
