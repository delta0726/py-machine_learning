# ******************************************************************************
# Title       : AI・データサイエンスのための数学プログラミング
# Chapter     : 2 機械学習を使った分析を行ってみよう
# Theme       : 2-6 決定木によって行動の原因を推定
# Creat Date  : 2021/12/20
# Final Update: 2022/09/07
# Page        : P76 - P79
# ******************************************************************************


# ＜概要＞
# - 決定木はツリーモデルの最も単純な形であり、予測や要因分析などに活用することができる
#   --- 決定木は可視化が可能なので、要因分析の文脈で使われることが多い


# ＜目次＞
# 0 準備
# 1 特徴量ベクトルの作成
# 2 k-meansによるクラスタリング
# 3 ラベルの作成
# 4 決定木によるモデル構築
# 5 決定木の可視化


# 0 準備 ---------------------------------------------------------------------

# ライブラリ
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from dtreeviz.trees import dtreeviz


# データ読み込み
df_info = pd.read_csv("csv/accomodation_info.csv", index_col=0, parse_dates=[0])

# データ確認
df_info


# 1 特徴量ベクトルの作成 ------------------------------------------------------------

# ＜ポイント＞
# - 利用回数上位の顧客ごとの月間利用回数を特徴量ベクトルとする


# インデックスの取得
x_0 = df_info.resample('M').count().drop(df_info.columns.values, axis=1)

# パラメータ設定
# --- 対象人数の設定
num = 100

# 配列の準備
list_vector = []

# 特徴量ベクトルの作成
# --- 顧客IDの抽出
# --- 月ごとの利用回数を特徴量として抽出
# --- 欠損値があった場合の穴埋め
# --- 特徴ベクトルとして追加
i_rank = 0
for i_rank in range(num):
    i_id = df_info['顧客ID'].value_counts().index[i_rank]
    df_i = df_info.loc[lambda x: x['顧客ID'] == i_id].filter(['顧客ID'])
    x_i = df_i.resample('M').count()
    x_i = pd.concat([x_0, x_i], axis=1).fillna(0)
    list_vector.append(x_i.iloc[:, 0].values.tolist())

# 特徴量ベクトルの変換
features = np.array(list_vector)

# データ確認
print(features)


# 2 k-meansによるクラスタリング ----------------------------------------------------

# モデル構築
# --- クラスタの予測
model = KMeans(n_clusters=4, random_state=0)
model.fit(features)

# モデル確認
vars(model)

# クラスタの取得
pred_class = model.predict(features)

# 要素数の確認
pd.DataFrame(pred_class).groupby(0).size()


# 3 ラベルの作成 ------------------------------------------------------------------

# ＜ポイント＞
# - クラスタリングで分類したグループの1つか否かを判定する


# パラメータ設定
# --- 分析したいクラス
target_class = 1

# ラベルの作成
# --- 分析対象のラベルならTrue
num = len(pred_class)
data_o = np.zeros(num)
for i in range(num):
    if pred_class[i] == target_class:
        data_o[i] = True
    else:
        data_o[i] = False

# データ確認
print(data_o)
pd.DataFrame(data_o).groupby(0).size()


# 4 決定木によるモデル構築 --------------------------------------------------------

# モデル構築
# --- 決定木による分類器
clf = DecisionTreeClassifier(max_depth=2)
clf = clf.fit(X=features, y=data_o)

# 確認
vars(clf)


# 5 決定木の可視化 ---------------------------------------------------------------

# indexの抽出
time_index = df_info.resample('M').count().index

# 決定木を描画
viz = dtreeviz(
    clf,
    features,
    data_o,
    target_name='Class',
    feature_names=time_index,
    class_names=['False', 'True'],
)

# 出力
viz
