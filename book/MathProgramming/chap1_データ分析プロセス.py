# ******************************************************************************
# Title       : AI・データサイエンスのための数学プログラミング
# Chapter     : 1 データを手にしてまず行うべきこと
# Theme       : 1-1 to 1-10 データ分析プロセス
# Creat Date  : 2021/12/18
# Final Update: 2022/09/01
# Page        : P28 - P57
# ******************************************************************************


# ＜概要＞
# - 宿泊プランのデータを分析してデータ分析プロセスを学ぶ


# ＜目次＞
# 0 準備
# 1 時系列データを可視化
# 2 基本統計量の算出
# 3 ヒストグラムの確認
# 4 分布の近似曲線の作成
# 5 プランごとにデータ抽出
# 6 大口顧客の行動分析
# 7 コロナ感染症前後の行動分析
# 8 条件による顧客の分類
# 9 条件にあった顧客をリストアップ


# 0 準備 ---------------------------------------------------------------------------

# ライブラリ
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import datetime as dt
import seaborn as sns


# データ読み込み
df_info = pd.read_csv("csv/accomodation_info.csv", index_col=0, parse_dates=[0])

# データ確認
df_info
df_info.info()


# 1 時系列データを可視化 ------------------------------------------------------------

# ＜ポイント＞
# - データの大まかなイメージを掴むため月次売上高の推移の可視化を行う
# - resample()はインデックスに対してグループ化を行う
#   --- sum()などの集計メソッドを適用する


# 月次売上高の集計
# --- 月間合計
sales_m = df_info['金額'].resample('M').sum()

# プロット作成
# --- 月次売上高は2020/03のコロナ蔓延化を境に激減している
plt.plot(sales_m, color='k')
plt.xticks(rotation=20)
plt.show()

# 月次利用者数の集計
count_m = df_info['顧客ID'].resample('M').count()

# プロット作成
# --- 月次利用者数も月次売上高と同様に激減している
plt.plot(count_m, color='k')
plt.xticks(rotation=20)
plt.show()


# 2 基本統計量の算出 -------------------------------------------------------

# ＜ポイント＞
# - 分析する系列ごとの統計量はあらかじめ確認しておく
#   --- 今回はメインの数値列は｢金額｣のみ


# ユニークIDの数
# --- 顧客IDは重複がある
# --- ユニークID：5486  全ID：71722
df_info['顧客ID'].value_counts().shape[0]
df_info['顧客ID'].count()

# 統計量の算出
# --- 顧客IDごとの宿泊回数の統計量
# --- 特定の顧客の宿泊回数が極端に多い
x_mean = df_info['顧客ID'].value_counts().mean()
x_median = df_info['顧客ID'].value_counts().median()
x_min = df_info['顧客ID'].value_counts().min()
x_max = df_info['顧客ID'].value_counts().max()

# 確認
print("平均値:", x_mean)
print("中央値:", x_median)
print("最小値", x_min)
print("最大値", x_max)


# 3 ヒストグラムの確認 ----------------------------------------------------

# ＜ポイント＞
# - ヒストグラムはデータ分布を知るために確認しておく
# - 顧客ごとの宿泊回数はべき乗分布となっている（ごくまれに非常に多く宿泊する人がいる）
#   --- ビジネスの現場でよく見られる分布
#   --- ｢パレート法則｣や｢80:20の法則｣と呼ばれ、売上の8割は2割の顧客が生み出している


# 顧客ごとの宿泊回数
x = df_info['顧客ID'].value_counts()

# ヒストグラムの作成
# --- x_hist: 頻度
# --- t_hist: 頻度の区分
x_hist, t_hist, _ = plt.hist(x, 21, color="k")
plt.show()

# ヒストグラムの作成（seaboarn）
sns.distplot(x, bins=13, color='#123456', label='data', kde=False, rug=False)
plt.show()


# 4 分布の近似曲線の作成 ----------------------------------------------------

# ＜ポイント＞
# - ヒストグラムの近似曲線を作成する
# - 近似曲線に関するパラメータを算出したものに基づいて描画する


# パラメータ設定
epsiron = 1
num = 15

# 変数定義
# --- 頻度をウエイトとして用いる
# --- 区分数のゼロ配列を作成（勾配計算に利用する）
weight = x_hist[1:num]
t = np.zeros(len(t_hist) - 1)

# 勾配計算
# --- 直前の区分値との傾きを算出
i = 1
for i in range(len(t)):
    t[i] = (t_hist[i] + t_hist[i + 1]) / 2

# 確認
print(t)

# パラメータの算出
# --- 最小二乗近似によるフィッティング
a, b = np.polyfit(x=t[1:num], y=np.log(x_hist[1:num]), deg=1, w=weight)

# フィッティング曲線（直線）の描画
xt = np.zeros(len(t))
for i in range(len(t)):
    xt[i] = a * t[i] + b

# 確認
print(xt)

# プロット作成
plt.plot(t_hist[1:], np.log(x_hist + epsiron), marker=".", color="k")
plt.plot(t, xt, color="r")
plt.show()


t = t_hist[1:]
xt = np.zeros(len(t))
for i in range(len(t)):
    xt[i] = math.exp(a * t[i] + b)

# 確認
print(xt)

# プロット作成
plt.bar(t_hist[1:], x_hist, width=8, color="k")
plt.plot(t, xt, color="r")
plt.show()


# 5 プランごとにデータ抽出 -------------------------------------------------

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
df_a = df_info[lambda x: x['プラン'] == 'A']
df_b = df_info[lambda x: x['プラン'] == 'B']
df_c = df_info[lambda x: x['プラン'] == 'C']
df_d = df_info[lambda x: x['プラン'] == 'D']

# ヒストグラム作成
# --- それぞれがべき乗分布に従っている
plot_hist(df_a)
plot_hist(df_b)
plot_hist(df_c)
plot_hist(df_d)


# 時系列データ作成
# --- プランごとの毎月の利用者
ts_a = df_a['顧客ID'].resample('M').count()
ts_b = df_b['顧客ID'].resample('M').count()
ts_c = df_c['顧客ID'].resample('M').count()
ts_d = df_d['顧客ID'].resample('M').count()

# 時系列プロット
#   --- プランB/Dは夕食付プランなので急落している
#   --- プランA/Cは夕食なしプランなのでテレワーク顧客の獲得で水準を維持している
plt.plot(ts_a, color='b', label="A")
plt.plot(ts_b, color='g', label="B")
plt.plot(ts_c, color='r', label="C")
plt.plot(ts_d, color='k', label="D")
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=12)
plt.xticks(rotation=20)
plt.show()




# 6 大口顧客の行動分析 ----------------------------------------------------------

# ＜ポイント＞
# - サンプル数の多い顧客の月別動向を確認する
#   --- ｢80:20の法則｣によると大口顧客は売上高への寄与度が非常に高いため


# レコード抽出＆表示
# --- サンプル数の多い顧客のレコードを抽出
# --- サンプル数が多いので5つのみ
i = 0
for i in range(5):
    _id = df_info['顧客ID'].value_counts().index[i]
    print(df_info.loc[lambda x: x['顧客ID'] == _id])


# 時系列プロット作成
# --- サンプル数が多いので5つのみ
i = 0
for i in range(5):
    _id = df_info['顧客ID'].value_counts().index[i]
    df_i = df_info.loc[lambda x: x['顧客ID'] == _id]
    ts_i = df_i['顧客ID'].resample('M').count()
    plt.plot(ts_i)
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


# # 時系列プロット作成
# # --- ファセットを活用したプロット
# _id = df_info['顧客ID'].value_counts().head(5).index
# df_top = df_info[lambda x: x['顧客ID'].isin(_id)].resample('M').count()
# grid = sns.FacetGrid(df_info[lambda x: x['']], col="year", hue="year", col_wrap=4, size=5)
# grid.map(sns.pointplot, 'month', 'passengers')


# 7 コロナ感染症前後の行動分析 ------------------------------------------------------

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


# 8 条件による顧客の分類 ------------------------------------------------------

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


# 9 条件にあった顧客をリストアップ ---------------------------------------------

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

