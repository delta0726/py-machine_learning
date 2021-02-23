# ******************************************************************************
# Chapter   : 2 モデル構築前のワークフローと前処理
# Title     : 2-5 カテゴリ値の変数を操作する（Recipe13)
# Created by: Owner
# Created on: 2020/12/24
# Page      : P52 - P57
# ******************************************************************************

# ＜概要＞
# - カテゴリカルデータは価値の高いデータを提供するが、連続値ではないため扱いにくい


# ＜多重共線性との関係＞
# - One-Hotエンコーディングはアルゴリズムによっては多重共線性を生み出す
#   --- 全て0という選択肢が存在するため
#   --- 1つ要素を削除して変換する方法もある（ダミー変換）
#   --- アルゴリズムによっては問題とならないものもある（線形回帰など）


# ＜目次＞
# 0 準備
# 1 iris Speciesのエンコーディング
# 2 エンコーディングと疎データ
# 3 マルチクラス回帰の練習


# 0 準備 -------------------------------------------------------------------------

import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_extraction import DictVectorizer


# データ準備
iris = datasets.load_iris()

# データ構造
iris.keys()

# データ格納
X = iris.data
y = iris.target


# 1 iris Speciesのエンコーディング -------------------------------------------------------

# インスタンスの生成
cat_encoder = preprocessing.OneHotEncoder()

# データ変換
# --- One-Hot Encoding
# --- yの分類ラベルを変換する場合はLabelBinarizerクラスを使用すべき（scikit-learn 0.20.2ドキュメント）
cat_encoder.fit_transform(y.reshape(-1, 1)).toarray()[:5]


# 2 エンコーディングと疎データ ------------------------------------------------------------

np.ones((3, 1)).shape

#
cat_encoder.transform(np.ones((3, 1))).toarray()


# 3 マルチクラス回帰 ---------------------------------------------------------------------

# ＜ポイント＞
# - irisのラベルをマルチクラス回帰を用いて学習器から予測する
#   --- 学習器にはリッジ回帰を用いる


# インスタンス生成
# --- リッジ回帰
ridge_inst = Ridge()

# インスタンス生成
# --- 多出力回帰
# --- リッジ回帰のインスタンスを受取る
multi_ridge = MultiOutputRegressor(ridge_inst, n_jobs=-1)

# インスタンス生成
# --- One-Hot Encoding
cat_encoder = preprocessing.OneHotEncoder()

# ラベル変換
# --- y: Species
y_multi = cat_encoder.fit_transform(y.reshape(-1, 1)).toarray()

# データ分割
# --- 層別サンプリング(y)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y_multi, stratify=y, random_state=7)

# 学習
multi_ridge.fit(X_train, y_train)

# 予測
y_multi_pre = multi_ridge.predict(X_test)
y_multi_pre[:5]

# バイナリ変数に変換
y_multi_pred = preprocessing.binarize(y_multi_pre, threshold=0.5)
y_multi_pred[:5]

# モデル精度の評価-1
# --- テストデータを用いて評価
# --- データセット全体でAUCを算出
roc_auc_score(y_test, y_multi_pred)

# モデル精度の評価-2
# --- グループごとにAccuracyとAUCを算出
for i in range(0, 3):
    print("")
    print('Accuracy', 'flower:', i, accuracy_score(y_test[:, i], y_multi_pred[:, i]))
    print('AUC', 'flower:', i, roc_auc_score(y_test[:, i], y_multi_pred[:, i]))


# 4 DictVectorizerクラス -----------------------------------------------------------

# ＜ポイント＞
# - DictVectorizerクラスは文字列を直接エンコーディングすることが可能

# インスタンス生成
dv = DictVectorizer()

# 辞書の作成
my_dict = [{'species': iris.target_names[i]} for i in y]
my_dict[:5]

# One-Hotエンコーディング
dv.fit_transform(my_dict).toarray()[:5]
