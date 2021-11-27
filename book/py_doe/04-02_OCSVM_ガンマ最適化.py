# ******************************************************************************
# Title     : Pythonで学ぶ実験計画法
# Chapter   : 4 モデルの適用範囲
# Theme     : データ密度（One-Class Support Vector Machine ガンマ最適化）
# Date      : 2021/11/27
# Page      : P91 - P98
# ******************************************************************************


# ＜概要＞
# - データ密度をOCSVMで測定することによりADの判定に用いる
# - OCSVMのガンマをチューニングしてモデリングする


# ＜目次＞
# 0 準備
# 1 データ定義
# 2 データ分割
# 3 ハイパーパラメータのチューニング
# 4 OCSVMによるADモデル構築
# 5 訓練データのデータのAD算出
# 6 予測データのデータのAD算出


# 0 準備 ----------------------------------------------------------------

# ライブラリ
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM

# データ準備
df = pd.read_csv('csv/resin.csv', index_col=0, header=0)
x_pred = pd.read_csv('csv/resin_prediction.csv', index_col=0, header=0)


# 1 データ定義 ------------------------------------------------------------

# ＜ポイント＞
# - 目的変数はpropertyとして、その他のデータを説明変数とする


# データ定義
# --- 目的変数
# --- 説明変数
y = df.iloc[:, 0]
x = df.iloc[:, 1:]


# 2 データ加工 ---------------------------------------------------------------

# ゼロ・バリアンス・フィルタ
deleting_variables = x.columns[x.std() == 0]
x = x.drop(deleting_variables, axis=1)
x_pred = x_pred.drop(deleting_variables, axis=1)

# データの標準化
autoscaled_x = (x - x.mean()) / x.std()
autoscaled_x_pred = (x_pred - x.mean()) / x.std()
autoscaled_y = (y - y.mean()) / y.std(ddof=1)


# 3 ハイパーパラメータのチューニング ---------------------------------------------

# ＜ポイント＞
# - ガウシアンカーネルのガンマを分散最大化の基準で最適化


# パラメータ設定
# --- γ
ocsvm_gammas = 2 ** np.arange(-20, 11, dtype=float)

# γの最適化
# --- 分散最大化
variance_of_gram_matrix = []
autoscaled_x_array = np.array(autoscaled_x)
for nonlinear_svr_gamma in ocsvm_gammas:
    gram_matrix = np.exp(- nonlinear_svr_gamma * ((autoscaled_x_array[:, np.newaxis] - autoscaled_x_array) ** 2).sum(axis=2))
    variance_of_gram_matrix.append(gram_matrix.var(ddof=1))

# γの最適値
optimal_gamma = \
    ocsvm_gammas[np.where(variance_of_gram_matrix == np.max(variance_of_gram_matrix))[0][0]]

# 最適化された γ
print('最適化された gamma :', optimal_gamma)


# 4 OCSVMによるADモデル構築 ---------------------------------------------------

# パラメータ設定
# --- サポートベクターの数の下限の割合
ocsvm_nu = 0.04

# インスタンス生成
# --- ガウシアンカーネル
ad_model = OneClassSVM(kernel='rbf', gamma=optimal_gamma, nu=ocsvm_nu)

# モデル構築
ad_model.fit(autoscaled_x)


# 5 訓練データのデータのAD算出 ------------------------------------------------------------

# データ密度
data_density_train = ad_model.decision_function(autoscaled_x)

# サポートベクトルの数
number_of_support_vectors = len(ad_model.support_)

# 外れ値のサンプル
# --- 個数
# --- 割合
number_of_outliers_in_training_data = sum(data_density_train < 0)
number_of_outliers_in_training_data / x.shape[0]

# データフレーム格納
data_density_train = \
    pd.DataFrame(data_density_train, index=x.index, columns=['ocsvm_data_density'])

# ADの判定
inside_ad_flag_train = data_density_train >= 0
inside_ad_flag_train.columns = ['inside_ad_flag']

# 確認
pd.concat([data_density_train, inside_ad_flag_train], axis=1)


# 6 予測データのデータのAD算出 ------------------------------------------------------------

# データ密度
data_density_pred = ad_model.decision_function(autoscaled_x_pred)

# 外れ値のサンプル
# --- 個数
number_of_outliers_in_prediction_data = sum(data_density_pred < 0)
number_of_outliers_in_prediction_data / x_pred.shape[0]

# データフレーム格納
data_density_pred = \
    pd.DataFrame(data_density_pred, index=x_pred.index, columns=['ocsvm_data_density'])

# ADの判定
inside_ad_flag_pred = data_density_pred >= 0
inside_ad_flag_pred.columns = ['inside_ad_flag']

# 確認
pd.concat([data_density_pred, inside_ad_flag_pred], axis=1)
