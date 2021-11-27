# ******************************************************************************
# Title     : Pythonで学ぶ実験計画法
# Chapter   : 5 実験計画法・適応的実験計画法の実践
# Theme     : 実験候補の生成
# Date      : 2021/11/27
# Page      : P103 - P105
# ******************************************************************************


# ＜概要＞
# - 実験のシミュレーションを行うためのデータセットを乱数から生成する
# - 乱数から実際のデータの範囲や特性をなるべく正確に表現する


# ＜目次＞
# 0 準備
# 1 実験用データの生成
# 2 制約条件によるデータの精緻化
# 3 最終データの確認

# 0 準備 ----------------------------------------------------------------------

# ライブラリ
import pandas as pd
import numpy as np
from numpy import matlib

# データロード
setting_of_generation = pd.read_csv('csv/setting_of_generation.csv', index_col=0, header=0)


# 1 実験用データの生成 ---------------------------------------------------------

# ＜ポイント＞
# - 実験用データを乱数から生成する
# - 実データのスケールが一致するようにリスケーリングする


# パラメータ生成
# --- 生成するサンプル数
# --- 合計を指定する特徴量がある場合の合計の値
number_of_generating_samples = 10000
desired_sum_of_components = 1

# サンプル生成
# --- 0-1の一様乱数
x_generated = np.random.rand(number_of_generating_samples, setting_of_generation.shape[1])

# データレンジの変換
# --- 特徴量ごとのレンジの上下限
# --- 生成した一様乱数を設定したレンジ内に分布させる
x_upper = setting_of_generation.iloc[0, :]
x_lower = setting_of_generation.iloc[1, :]
x_generated = x_generated * (x_upper.values - x_lower.values) + x_lower.values


# 2 制約条件によるデータの精緻化 --------------------------------------------------

# ＜ポイント＞
# - 実データで分かっている性質があれば反映する


# 合計値の制約
# --- 合計を desired_sum_of_components にする特徴量がある場合
if setting_of_generation.iloc[2, :].sum() != 0:
    for group_number in range(1, int(setting_of_generation.iloc[2, :].max()) + 1):
        variable_numbers = np.where(setting_of_generation.iloc[2, :] == group_number)[0]
        actual_sum_of_components = x_generated[:, variable_numbers].sum(axis=1)
        actual_sum_of_components_converted = np.matlib.repmat(np.reshape(actual_sum_of_components, (x_generated.shape[0], 1)) , 1, len(variable_numbers))
        x_generated[:, variable_numbers] = x_generated[:, variable_numbers] / actual_sum_of_components_converted * desired_sum_of_components
        deleting_sample_numbers, _ = np.where(x_generated > x_upper.values)
        x_generated = np.delete(x_generated, deleting_sample_numbers, axis=0)
        deleting_sample_numbers, _ = np.where(x_generated < x_lower.values)
        x_generated = np.delete(x_generated, deleting_sample_numbers, axis=0)

# 数値の丸め込みをする場合
if setting_of_generation.shape[0] >= 4:
    for variable_number in range(x_generated.shape[1]):
        x_generated[:, variable_number] = np.round(x_generated[:, variable_number],
                                                   int(setting_of_generation.iloc[3, variable_number]))


# 3 最終データの確認 -------------------------------------------------------------

# データフレーム格納
x_generated = pd.DataFrame(x_generated, columns=setting_of_generation.columns)

# データ確認
x_generated
