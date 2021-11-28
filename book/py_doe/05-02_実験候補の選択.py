# ******************************************************************************
# Title     : Pythonで学ぶ実験計画法
# Chapter   : 5 実験計画法・適応的実験計画法の実践
# Theme     : 実験候補の選択
# Date      : 2021/11/27
# Page      : P105 - P107
# ******************************************************************************


# ＜概要＞
# - 乱数で作成したデータセットから最初に実験するデータ群を指定する
# - 類似サンプルが少なくなるようにD最適化基準が最大化するようにサンプルを選択する


# ＜目次＞
# 0 準備
# 1 実験サンプルの選択
# 2 データ確認
# 3 データ保存


# 0 準備 ---------------------------------------------------------------------------

# ライブラリ
import pandas as pd
import numpy as np


# データロード
# --- [9709 rows x 5 columns]
x_generated = pd.read_csv('csv/generated_samples.csv', index_col=0, header=0)


# 1 実験サンプルの選択 ----------------------------------------------------------------

# ＜ポイント＞
# - 選択されたセットの中に偶然に類似したサンプルが存在してしまう可能性がある
#   --- ランダムに抽出したデータからD最適化基準を計算する
#   --- D最適化基準が最大となる結果を抽出して初期サンプルとする


# パラメータ設定
# --- 選択するサンプル数
# --- ランダムにサンプルを選択してD最適基準を計算する繰り返し回数
number_of_selecting_samples = 30
number_of_random_searches = 1000

# 実験条件の候補のインデックスの作成
# --- 元データのレコード数（9709）
all_indexes = list(range(x_generated.shape[0]))


# D最適基準に基づくサンプル選択
np.random.seed(11)
random_search_number = 0
for random_search_number in range(number_of_random_searches):
    # 1. ランダムに候補を選択
    new_indexes = np.random.choice(all_indexes, number_of_selecting_samples, replace=False)
    new_samples = x_generated.iloc[new_indexes, :]

    # 2. D最適基準を計算
    autoscaled_new_samples = (new_samples - new_samples.mean()) / new_samples.std()
    xt_x = np.dot(autoscaled_new_samples.T, autoscaled_new_samples)
    d_optimal_value = np.linalg.det(xt_x)

    # 3. D最適基準が前回までの最大値を上回ったら、選択された候補を更新
    if random_search_number == 0:
        best_d_optimal_value = d_optimal_value.copy()
        selected_sample_indexes = new_indexes.copy()
    else:
        if best_d_optimal_value < d_optimal_value:
            best_d_optimal_value = d_optimal_value.copy()
            selected_sample_indexes = new_indexes.copy()

# リスト型に変換
selected_sample_indexes = list(selected_sample_indexes)


# 2 データ確認 ----------------------------------------------------------------

# データ格納
# --- 選択されたサンプル
# --- 選択されなかったサンプル
selected_samples = x_generated.iloc[selected_sample_indexes, :]
remaining_indexes = np.delete(all_indexes, selected_sample_indexes)
remaining_samples = x_generated.iloc[remaining_indexes, :]

# 相関行列の確認
# --- SciView()
corr = selected_samples.corr()
print(round(corr, 2))


# 3 データ保存 ----------------------------------------------------------------

# 保存
# --- 選択されたサンプル
# --- 選択されなかったサンプル
# selected_samples.to_csv('csv/selected_samples.csv')
# remaining_samples.to_csv('csv/remaining_samples.csv')
