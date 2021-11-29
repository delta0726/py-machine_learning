# ***************************************************************************************
# Title     : 意思決定分析と予測の活用
# Chapter   : 第2部 決定分析の基本
# Theme     : 第3章 決定分析におけるPythonの活用
# Created on: 2021/11/30
# Page      : P44 - P56 / P65 - P79
# ***************************************************************************************


# ＜概要＞
# - ｢利益=収益-費用｣として算出されるように、効用(利得)はコスト控除後で決定される
# - 利得行列は収益と費用を行列形式で計算したものから算出される
#   --- 利得行列は対面するタスクの収益とコストを明確して個別に定義していく必要がある
#   --- 行列表記することで条件が変わった際の変化が明確に分かる


# ＜5つの決定基準＞
# - マキシマックス基準（Maximax）
# - マキシミン基準（Maximin）
# - ハーヴィッツ基準（Hurwicz）
# - ミニマックスリグレット基準（Minimax Regret）
# - ラプラス基準（Laplace）


# ＜目次＞
# 0 準備
# 1 製品製造個数の定義
# 2 売上行列と製造コスト
# 3 利得行列の算出
# 4 利得行列を作成する関数
# 5 マキシマックス基準
# 6 マキシミン基準
# 7 ハーヴィッツ基準
# 8 ミニマックスリグレット基準
# 9 ラプラス基準
# 10 感応度分析


# 0 準備 -------------------------------------------------------------------------------

# 数値計算に使うライブラリ
import numpy as np
import pandas as pd


# 1 製品製造個数の定義 --------------------------------------------------------------------

# 製品個数のパラメータ
# --- 機械1台で作られる製品数(個)
# --- 好況時の需要量(個)
# --- 不況時の需要量(個)
machine_ability = 5000
demand_boom = 10000
demand_slump = 5000

# 出荷される製品の個数
num_product_df = pd.DataFrame({
    '0台': [min([machine_ability * 0, demand_boom]),
            min([machine_ability * 0, demand_slump])],
    '1台': [min([machine_ability * 1, demand_boom]),
            min([machine_ability * 1, demand_slump])],
    '2台': [min([machine_ability * 2, demand_boom]),
            min([machine_ability * 2, demand_slump])]
}, index = ['好況', '不況'])

# 確認
print(num_product_df)


# 2 売上行列と製造コスト ---------------------------------------------

# ＜ポイント＞
# - 売上と製造コストは好況/不況に限らず一定


# 売上のパラメータ
# --- 製品1つの販売価格(万円)
sale_price = 0.2

# 売上行列
sales_df = num_product_df * sale_price
print(sales_df)


# 製造コストのパラメータ
# --- 工場の固定費用(万円)
# --- 機械1台の稼働コスト(万円)
fixed_cost = 100
run_cost = 600

# 製造コスト
run_cost_df = pd.DataFrame({
    '0台': np.repeat(fixed_cost + run_cost * 0, 2),
    '1台': np.repeat(fixed_cost + run_cost * 1, 2),
    '2台': np.repeat(fixed_cost + run_cost * 2, 2)
}, index = ['好況', '不況'])
print(run_cost_df)


# 3 利得行列の算出 ----------------------------------------------------

# ＜ポイント＞
# - 利得行列は売上行列から製造コストを差し引いて算出する


# 利得行列
payoff_df = sales_df - run_cost_df
print(payoff_df)


# 4 利得行列を作成する関数 ---------------------------------------------

# 関数定義
def calc_payoff_table(fixed_cost, run_cost, sale_price,
                      machine_ability, demand_boom, demand_slump):
    # 数量行列
    num_product_df = pd.DataFrame({
        '0台': [min([machine_ability * 0, demand_boom]),
                min([machine_ability * 0, demand_slump])],
        '1台': [min([machine_ability * 1, demand_boom]),
                min([machine_ability * 1, demand_slump])],
        '2台': [min([machine_ability * 2, demand_boom]),
                min([machine_ability * 2, demand_slump])]
    })
    # 売上行列
    sales_df = num_product_df * sale_price
    # 製造コスト行列
    run_cost_df = pd.DataFrame({
        '0台': np.repeat(fixed_cost + run_cost * 0, 2),
        '1台': np.repeat(fixed_cost + run_cost * 1, 2),
        '2台': np.repeat(fixed_cost + run_cost * 2, 2)
    })
    # 利得行列
    payoff_df = sales_df - run_cost_df
    payoff_df.index = ['好況', '不況']
    # 結果を返す
    return(payoff_df)


# 利得行列の計算
payoff = calc_payoff_table(fixed_cost=100, run_cost=600, sale_price=0.2,
                           machine_ability=5000, demand_boom=10000,
                           demand_slump=5000)
print(payoff)


# 5 マキシマックス基準 --------------------------------------------------------------------------

# ＜ポイント＞
# - 全てのケースで最も最大の利得を効用とする
#   --- 行列の最大値を求めればよい（複数存在するケースがある点に注意）


# 5-1 効用の算出 -------------------------------------------------------

# 選択肢ごとの最大利得
# --- 好況/不況に限らず最大値を取る
payoff.max()

# 効用（マキシマックス基準）
payoff.max().max()

# 稼働台数
# --- 最大値のインデックスを取得
payoff.max().idxmax()


# 5-2 最大値と同じ効用を持つケースをリスト化 --------------------------------

# 最大値と等しい利得を持つ要素を取得する
payoff.max()[payoff.max() == payoff.max().max()]

# 最大値のインデックスだけを取得する
list(payoff.max()[payoff.max() == payoff.max().max()].index)


# 5-3 関数定義 ---------------------------------------------------------

# 最大値をとるインデックスを取得
# --- 最大値が複数ある場合はすべて出力する
def argmax_list(series):
    return(list(series[series == series.max()].index))

# 最小値をとるインデックスを取得
# --- 最小値が複数ある場合はすべて出力
def argmin_list(series):
    return(list(series[series == series.min()].index))

# 動作確認
print('Maximax:', argmax_list(payoff.max()))


# 6 マキシミン基準 -------------------------------------------------------

# ＜ポイント＞
# - 選択肢ごとのネガティブシナリオ(不況)を考慮する
#   --- その中で最大の効用を選択する（効用最大化）

# 選択肢ごとの最小利得
payoff.min()

# 効用（マキシミン基準）
payoff.min().max()

# 稼働台数
print('Maximin:', argmax_list(payoff.min()))


# 7 ハーヴィッツ基準 -----------------------------------------------------

# ＜ポイント＞
# - 楽観係数(alpha)でマキシマックス基準とマキシミン基準を加重平均したもの

# ハーヴィッツの基準
# ---- alpha=0.6
hurwicz = payoff.max() * 0.6 + payoff.min() * (1 - 0.6)
hurwicz

# 効用（ハーヴィッツ基準）
hurwicz.max()

# 稼働台数
# --- 1台と2台は同等に好ましいという結果
argmax_list(hurwicz)


# 8 ミニマックスリグレット基準 -----------------------------------------------

# ＜ポイント＞
# - ｢後悔(リグレット)の大きさの最大値｣を最小化する基準
# - リグレットは機会損失の大きさを指す

# 機会最大利得
# --- 「最も利得が高くなる選択」をとったときの利得
# --- 利得行列と同じ大きさの行列を生成
best_df = pd.concat([payoff.max(axis=1)] * payoff.shape[1], axis=1)
best_df.columns = payoff.columns
print(best_df)

# リグレット
# --- 機会費用（費用なのでプラス表記）
regret_df = best_df - payoff
print(regret_df)

# リグレットの最大値
# --- 各々の選択肢ごと
regret_df.max()

# リグレット最大の選択肢
argmin_list(regret_df.max())


# 9 ラプラス基準 ---------------------------------------------------------

# ＜ポイント＞
# - 選択肢ごとに利得の算術平均を計算して、その後に最大値を取る選択肢
#   --- ハーヴィッツ基準の楽観係数(alpha)が0.5のケース


# 選択肢ごとの利得の平均値
payoff.mean()

print('Laplace:', argmax_list(payoff.mean()))


# 10 感応度分析 ---------------------------------------------------------

# ＜ポイント＞
# - 感応度分析ではパラメータを変化させた場合に結果(意思決定)がどのように変化するかを確認する
# - ミニマックスリグレット基準で評価（機会損失を最小化）


# ミニマックスリグレット基準による決定を行う関数-----------------

# 関数定義
def minimax_regret(payoff_table):
    best_df = pd.concat(
        [payoff_table.max(axis=1)] * payoff_table.shape[1], axis=1)
    best_df.columns = payoff_table.columns
    regret_df = best_df - payoff_table
    return(argmin_list(regret_df.max()))

# 稼働台数
print('Minimax regret:', minimax_regret(payoff))


# 機械1台の稼働コストを増やした ---------------------------

# ペイオフテーブルの作成
# --- run_costを600から625に変更
payoff_2 = calc_payoff_table(fixed_cost=100, run_cost=625, sale_price=0.2,
                             machine_ability=5000, demand_boom=10000,
                             demand_slump=5000)

# 確認
print(payoff_2)

# 稼働台数
print('Minimax regret:', minimax_regret(payoff_2))


# 機械1台の稼働コストを減らした ---------------------------

# ペイオフテーブルの作成
# --- run_costを600から575に変更
payoff_3 = calc_payoff_table(fixed_cost=100, run_cost=575, sale_price=0.2,
                             machine_ability=5000, demand_boom=10000,
                             demand_slump=5000)

# 確認
print(payoff_3)

# 稼働台数
print('Minimax regret:', minimax_regret(payoff_3))


# 機械1台の稼働コストをさらに減らした -------------------------

# ペイオフテーブルの作成
# --- run_costを600から575に変更
payoff_4 = calc_payoff_table(fixed_cost=100, run_cost=500, sale_price=0.2,
                             machine_ability=5000, demand_boom=10000,
                             demand_slump=5000)

# 確認
print(payoff_4)

# 稼働台数
print('Minimax regret:', minimax_regret(payoff_4))
