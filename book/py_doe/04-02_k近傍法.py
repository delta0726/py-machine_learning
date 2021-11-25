# ******************************************************************************
# Title     : Pythonで学ぶ実験計画法
# Chapter   : 4 モデルの適用範囲
# Theme     : データ密度（k近傍法）
# Date      : 2021/11/25
# Page      : P89 - P91
# ******************************************************************************


# ＜概要＞
# - データ分布が複数領域に分かれている場合でも適切にADを設定することができる
# - k近傍法は複数の特徴量に対する距離が最も近いk個のサンプルを選択して平均距離をデータ密度とする
#   --- パラメータに基づいてADの範囲を決定する


# ＜目次＞
# 0 準備
# 1 データ定義
# 2 データ分割
# 3 モデル構築
# 4 距離データの取得
# 5 k近傍法によるAD
# 6 予測用データのAD


# 0 準備 ----------------------------------------------------------------

# ライブラリ
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# データ準備
df = pd.read_csv('csv/resin.csv', index_col=0, header=0)
x_pred = pd.read_csv('csv/resin_pred.csv', index_col=0, header=0)


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


# 3 モデル構築 ----------------------------------------------------------

# パラメータ設定
k_in_knn = 5

# インスタンス生成
ad_model = NearestNeighbors(n_neighbors=k_in_knn, metric='euclidean')

# モデル構築
ad_model.fit(autoscaled_x)

# 確認
vars(ad_model)


# 4 距離データの取得 ----------------------------------------------------------

# 距離の出力
# --- サンプルごとのk最近傍との距離（インデックスも出力）
knn_distance_train, knn_index_train = \
    ad_model.kneighbors(autoscaled_x, n_neighbors=k_in_knn + 1)

# データフレーム格納
knn_distance_train = pd.DataFrame(knn_distance_train, index=autoscaled_x.index)

# 平均距離の算出
# 自分以外の k_in_knn 個の距離の平均
mean_of_knn_distance_train = \
    pd.DataFrame(knn_distance_train.iloc[:, 1:].mean(axis=1),
                 columns=['mean_of_knn_distance'])


# 5 k近傍法によるAD ----------------------------------------------------------

# パラメータ設定
rate_of_training_samples_inside_ad = 0.96

# 距離の平均の小さい順に並び替え
sorted_mean_of_knn_distance_train = \
    mean_of_knn_distance_train.iloc[:, 0].sort_values(ascending=True)

# 閾値の設定
# --- トレーニングデータのサンプルの rate_of_training_samples_inside_ad * 100 % が含まれるようにしきい値を設定
ad_threshold = \
    sorted_mean_of_knn_distance_train.iloc[round(autoscaled_x.shape[0]
                                                 * rate_of_training_samples_inside_ad) - 1]

# AD内外の判定
# --- トレーニングデータ
inside_ad_flag_train = mean_of_knn_distance_train <= ad_threshold
inside_ad_flag_train.columns = ['inside_ad_flag']


# 6 予測用データのAD -------------------------------------------------------------

# k近傍の距離の計算
# --- 予測用データ
knn_dist_pred, knn_index_pred = ad_model.kneighbors(autoscaled_x_pred)
knn_dist_pred = pd.DataFrame(knn_dist_pred, index=x_pred.index)

# k_in_knn 個の距離の平均
mean_of_knn_dist_pred = \
    pd.DataFrame(knn_dist_pred.mean(axis=1), columns=['mean_of_knn_distance'])

# 予測用データに対して、AD の中か外かを判定
inside_ad_flag_pred = mean_of_knn_dist_pred <= ad_threshold
inside_ad_flag_pred.columns = ['inside_ad_flag']
