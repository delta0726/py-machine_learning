# ******************************************************************************
# Title       : AI・データサイエンスのための数学プログラミング
# Chapter     : 3 必要なデータ数を検討しよう
# Theme       : 3-6 宿泊者数との相関関係を仮定して被害額の推移を推測
# Creat Date  : 2021/12/24
# Final Update:
# Page        : P111 - P113
# ******************************************************************************


# ＜概要＞
# - データ加工の練習


# ＜目次＞
# 0 準備
# 1 データ集計
# 2 被害額の推移


# 0 準備 ---------------------------------------------------------------------

# ライブラリ
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

# データ準備
df_info = pd.read_csv("csv/accomodation_info.csv", index_col=0, parse_dates=[0])


# 1 データ集計 ----------------------------------------------------------------

# 一日あたりの被害額の平均値を設定
theft_per_day = 5880

# 一日あたりの宿泊者数の抽出
df_num = df_info.resample('D').count().iloc[:, 0]

# 1ヶ月分の宿泊者数
target_date = dt.datetime(2018, 11, 30)
df_num_201811 = df_num[df_num.index <= target_date]
print("1ヶ月の宿泊者数:", sum(df_num_201811))

# 平均宿泊者数
num_per_day = sum(df_num_201811) / len(df_num_201811)
print("1日あたりの平均宿泊者数:", num_per_day)

# 宿泊者1人あたりの平均被害額
theft_per_person = theft_per_day / num_per_day
print("宿泊者1人あたりの平均被害額:", theft_per_person)


# 2 被害額の推移 ---------------------------------------------------------------

# 配列生成
estimated_theft = np.zeros(len(df_num))

for i in range(len(df_num)):
    estimated_theft[i] = df_num.iloc[i] * theft_per_person


# データフレーム格納
df_estimated_theft = pd.DataFrame(estimated_theft, index=df_num.index,
                                  columns=["推定被害額"])

# 合計額の確認
print("二年間の推定被害総額:", sum(df_estimated_theft["推定被害額"]))

# プロット作成
plt.plot(df_estimated_theft, color="k")
plt.xticks(rotation=60)
plt.show()
