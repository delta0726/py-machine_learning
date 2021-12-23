# ******************************************************************************
# Title       : AI・データサイエンスのための数学プログラミング
# Chapter     : 3 必要なデータ数を検討しよう
# Theme       : 3-3 1ヵ月のデータを正確に抽出する
# Creat Date  : 2021/12/24
# Final Update:
# Page        : P101 - P104
# ******************************************************************************


# ＜概要＞
# - データ集計の練習


# ＜目次＞
# 0 準備
# 1 被害総額の計算：ループ
# 2 被害総額の計算：演算


# 0 準備 ---------------------------------------------------------------------

# ライブラリ
import pandas as pd


# データ取得
df_theft_201811 = pd.read_csv("csv/theft_list_201811.csv", index_col=0, parse_dates=[0])
df_amenity_price = pd.read_csv("csv/amenity_price.csv", index_col=0, parse_dates=[0])

# 確認
df_theft_201811
df_amenity_price


# 1 被害総額の計算：ループ --------------------------------------------------------

# ＜ポイント＞
# - 1セルずつループして金学区を集計
#   --- ループと複合代入演算子の掴型の確認


# 初期値の設定
total_amount = 0
total_theft = 0

# ループ集計
for i_index in range(len(df_theft_201811.index)):
    for i_column in range(len(df_theft_201811.columns)):
        total_amount += df_theft_201811.iloc[i_index, i_column] * df_amenity_price["金額"].iloc[i_column]
        total_theft += df_theft_201811.iloc[i_index, i_column]
        if df_theft_201811.iloc[i_index, i_column] > 0:
            print(df_theft_201811.index[i_index], df_theft_201811.columns[i_column],
                  df_theft_201811.iloc[i_index, i_column], "点")

# 確認
print("被害総額", total_amount, "円")
print("被害件数", total_theft, "件")


# 2 被害総額の計算：演算 --------------------------------------------------------

# ＜ポイント＞
# - 安易にループを使うよりもデータフレーム操作で集計するほうが効率がよい
#   --- ロング型で集計するのがセオリー


# 単価テーブル
df_price = df_amenity_price \
    .reset_index() \
    .set_axis(['アイテム', '単価'], axis=1)

# 数量テーブル
df_quantity = df_theft_201811 \
    .reset_index() \
    .melt(id_vars="日時", var_name="アイテム", value_name="個数")

# データ結合
df_theft_agg = pd.merge(df_quantity, df_price, on='アイテム') \
    .assign(Total=lambda x: x['個数'] * x['単価'])

# 確認
print("被害総額", df_theft_agg['Total'].sum(), "円")
print("被害件数", df_theft_agg['個数'].sum(), "件")
