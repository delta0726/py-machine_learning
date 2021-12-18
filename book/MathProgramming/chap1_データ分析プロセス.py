# ******************************************************************************
# Title       : AI・データサイエンスのための数学プログラミング
# Chapter     : 1 データを手にしてまず行うべきこと
# Theme       : 1-1 to 1-10 データ分析プロセス
# Creat Date  : 2021/12/18
# Final Update: 2021/12/18
# Page        : P28 - P57
# ******************************************************************************


# ＜概要＞
# - 宿泊プランのデータを分析する


# ＜目次＞
# 0 準備
# 1 データの読込
# 2 時系列データを可視化
# 3 基本統計量の算出
# 4 ヒストグラムの確認
# 5 分布の近似曲線の作成
# 6 プランごとにデータ抽出
# 7 大口顧客の行動分析
# 8 コロナ感染症前後の行動分析


# 0 準備 ---------------------------------------------------------------------------

# ライブラリ
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import datetime as dt

# 1 データの読込 --------------------------------------------------------------------

# データ読み込み
df_info = pd.read_csv("csv/accomodation_info.csv", index_col=0, parse_dates=[0])

# データ確認
df_info

# 2 時系列データを可視化 ------------------------------------------------------------

# ＜ポイント＞
# - データの大まかなイメージを掴むために可視化を行う


# 月次売上高の集計
# --- 月間合計
sales_m = df_info['金額'].resample('M').sum()

# プロット作成
plt.plot(sales_m, color='k')
plt.show()

# 月次利用者数の集計
count_m = df_info['顧客ID'].resample('M').count()

# プロット作成
plt.plot(count_m, color='k')
plt.xticks(rotation=20)
plt.show()

# 3 基本統計量の算出 -------------------------------------------------------

# ユニークIDの数
# --- 顧客IDは重複がある
df_info['顧客ID'].value_counts().shape[0]

# 統計量の算出
# --- 顧客IDごとの宿泊回数の統計量
x_mean = df_info['顧客ID'].value_counts().mean()
x_median = df_info['顧客ID'].value_counts().median()
x_min = df_info['顧客ID'].value_counts().min()
x_max = df_info['顧客ID'].value_counts().max()

# 確認
print("平均値:", x_mean)
print("中央値:", x_median)
print("最小値", x_min)
print("最大値", x_max)

# 4 ヒストグラムの確認 ----------------------------------------------------

# ＜ポイント＞
# - ヒストグラムはデータ分布を知るために確認しておく
# - 顧客ごとの宿泊回数はべき乗分布となっている（ごくまれに非常に多く宿泊する人がいる）


# 顧客ごとの宿泊回数
x = df_info['顧客ID'].value_counts()

# ヒストグラムの作成
# --- x_hist: カウント
# --- t_hist: 頻度の区分
x_hist, t_hist, _ = plt.hist(x, 21, color="k")
plt.show()

# 5 分布の近似曲線の作成 ----------------------------------------------------

# ＜ポイント＞
# - ヒストグラムの近似曲線を作成する
# - 近似曲線に関するパラメータを算出したものに基づいて描画する


# パラメータ設定
epsiron = 1
num = 15

# 変数定義
# --- 頻度をウエイトとして用いる
# --- 区分数のゼロ配列を作成
weight = x_hist[1:num]
t = np.zeros(len(t_hist) - 1)

#
i = 1
for i in range(len(t_hist) - 1):
    t[i] = (t_hist[i] / t_hist[i + 1]) / 2

# パラメータの算出
# --- 最小二乗近似によるフィッティング
a, b = np.polyfit(t[1:num], np.log(x_hist[1:num]), 1, w=weight)

# フィッティング曲線（直線）の描画
xt = np.zeros(len(t))
for i in range(len(t)):
    xt[i] = a * t[i] + b
plt.plot(t_hist[1:], np.log(x_hist + epsiron), marker=".", color="k")
plt.plot(t, xt, color="r")
plt.show()

t = t_hist[1:]
xt = np.zeros(len(t))
for i in range(len(t)):
    xt[i] = math.exp(a * t[i] + b)

plt.bar(t_hist[1:], x_hist, width=8, color="k")
plt.plot(t, xt, color="r")
plt.show()


# 6 プランごとにデータ抽出 -------------------------------------------------

# ＜ポイント＞
# - コロナ感染症でプランごとの推移を確認する
#   --- プランB/Dは夕食付プランなので急落している
#   --- プランA/Cは夕食なしプランなのでテレワーク顧客の獲得で水準を維持している


# 関数定義
# --- ヒストグラム作成
def plot_hist(df):
    x = df['顧客ID'].value_counts()
    xa_hist, ta_hist, _ = plt.hist(x=x, bins=21, color='k')
    plt.show()


# データ抽出
df_a = df_info.loc[lambda x: x['プラン'] == 'A']
df_b = df_info.loc[lambda x: x['プラン'] == 'B']
df_c = df_info.loc[lambda x: x['プラン'] == 'C']
df_d = df_info.loc[lambda x: x['プラン'] == 'D']

# ヒストグラム作成
plot_hist(df_a)
plot_hist(df_b)
plot_hist(df_c)
plot_hist(df_d)

# 時系列プロット
# --- プランごとの毎月の利用者
plt.plot(df_a['顧客ID'].resample('M').count(), color='b', label="A")
plt.plot(df_b['顧客ID'].resample('M').count(), color='g', label="B")
plt.plot(df_c['顧客ID'].resample('M').count(), color='r', label="C")
plt.plot(df_d['顧客ID'].resample('M').count(), color='k', label="D")
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=12)
plt.xticks(rotation=20)
plt.show()

# 7 大口顧客の行動分析 ----------------------------------------------------------

# ＜ポイント＞
# - サンプル数の多い顧客の月別動向を確認する


# レコード抽出
# --- サンプル数の多い顧客のレコードを抽出
# --- サンプル数が多いので5つのみ
for i in range(5):
    _id = df_info['顧客ID'].value_counts().index[i]
    print(df_info.loc[lambda x: x['顧客ID'] == _id])

# 時系列プロット作成
# --- サンプル数が多いので5つのみ
for i in range(5):
    id_ = df_info['顧客ID'].value_counts().index[i]
    df_i = df_info.loc[lambda x: x['顧客ID'] == id_]
    plt.plot(df_i['顧客ID'].resample('M').count())
    plt.xticks(rotation=20)

plt.show()

# 時系列プロット作成
# --- サンプル数が多いので5つのみ
for i in range(5, 10):
    id_ = df_info['顧客ID'].value_counts().index[i]
    df_i = df_info.loc[lambda x: x['顧客ID'] == id_]
    plt.plot(df_i['顧客ID'].resample('M').count())
    plt.xticks(rotation=20)

plt.show()


# 8 コロナ感染症前後の行動分析 ------------------------------------------------------

# ＜ポイント＞
# - データを2群に分けて散布図にして傾向の変化を確認する


# インデックス解除
df_info_2 = df_info.reset_index()

# データ分割
target_date = dt.datetime(2020, 3, 1)
df_info_pre = df_info_2.loc[lambda x: x['日時'] < target_date]
df_info_post = df_info_2.loc[lambda x: x['日時'] >= target_date]

# 確認
df_info_pre.shape[0]
df_info_post.shape[0]

# パラメータ設定
# --- 抽出ID数
num = 200

# 配列準備
count_pre_and_post = np.zeros((num, 2))

# 散布図データ作成
# --- コロナ前後の顧客ごとの宿泊回数を取得
# --- ループを使わなくてもフラグとグループ集計で対応可能（ここでは書籍どおりの記述）
i_rank = 0
for i_rank in range(num):
    id_ = df_info['顧客ID'].value_counts().index[i_rank]
    count_pre_and_post[i_rank][0] = int(df_info_pre[df_info_pre['顧客ID'] == id_].count()[0])
    count_pre_and_post[i_rank][1] = int(df_info_post[df_info_post['顧客ID'] == id_].count()[0])

# データラベルのプロット
# --- 見にくいため非表示
# for i_rank in range(num):
#     id_ = df_info['顧客ID'].value_counts().index[i_rank]
#     text = str(id_) + "(" + str(i_rank) + ")"
#     plt.text(count_pre_and_post[i_rank][0], count_pre_and_post[i_rank][1], text, color="k")

# プロット作成
plt.scatter(count_pre_and_post.T[0], count_pre_and_post.T[1], color="k")
plt.xlabel("pre epidemic")
plt.ylabel("post epidemic")
plt.show()


# 9 条件による顧客の分類 ------------------------------------------------------

# パラメータ設定
num = 200
threshold_post = 50

# 配列準備
count_pre_and_post = np.zeros((num, 2))

# 散布図データ作成
# --- コロナ前後の顧客ごとの宿泊回数を取得
for i_rank in range(num):
    id_ = df_info['顧客ID'].value_counts().index[i_rank]
    count_pre_and_post[i_rank][0] = int(df_info_pre[df_info_pre['顧客ID'] == id_].count()[0])
    count_pre_and_post[i_rank][1] = int(df_info_post[df_info_post['顧客ID'] == id_].count()[0])

# プロット表示
# --- データラベルは見にくいため非表示
for i_rank in range(num):
    if count_pre_and_post[i_rank][1] > threshold_post:
        temp_color = "r"
    else:
        temp_color = "k"
    plt.scatter(count_pre_and_post[i_rank][0], count_pre_and_post[i_rank][1], color=temp_color)


plt.xlabel("pre epidemic")
plt.ylabel("post epidemic")
plt.show()


# 10 条件にあった顧客をリストアップ ---------------------------------------------

# パラメータ設定
num = 200
threshold_post = 50

# 配列の生成
# --- 顧客リスト
list_id = []
list_name = []
list_date_pre = []
list_date_post = []
count_pre_and_post = np.zeros((num, 2))


for i_rank in range(num):
    id_ = df_info['顧客ID'].value_counts().index[i_rank]
    count_pre_and_post[i_rank][0] = int(df_info_pre[df_info_pre['顧客ID'] == id_].count()[0])
    count_pre_and_post[i_rank][1] = int(df_info_post[df_info_post['顧客ID'] == id_].count()[0])

for i_rank in range(num):
    id_ = df_info['顧客ID'].value_counts().index[i_rank]
    text = str(id_) + "(" + str(i_rank) + ")"
    if count_pre_and_post[i_rank][1] > threshold_post:
        list_id.append(id_)
        list_name.append(df_info['宿泊者名'][df_info['顧客ID'] == id].iloc[0])
        list_date_pre.append(count_pre_and_post[i_rank][0])
        list_date_post.append(count_pre_and_post[i_rank][1])

# リストをデータフレーム形式に変換
df = pd.DataFrame([list_id])
df = df.T
df.columns = ['顧客ID']
df['宿泊者名'] = list_name
df['宿泊日数（流行前）'] = list_date_pre
df['宿泊日数（流行後）'] = list_date_post
print(df)

