# ******************************************************************************
# Chapter   : 2 モデル構築前のワークフローと前処理
# Title     : 2-4 閾値化を通じて二値の特徴量を作成（Recipe12)
# Created by: Owner
# Created on: 2020/12/24
# Page      : P50 - P52
# ******************************************************************************

# ＜概要＞
# - 連続データの離散化を行う
#   --- データの表現力を意図的に落とすことが有効なシーンは多い
# - scikit-learnでは変換モジュールが2つ用意されている
#   --- processing.binarize()  : 関数
#   --- processing.Binarizer() : クラス(パイプライン)


# ＜目次＞
# 0 準備
# 1 閾値でバイナリ変換
# 2 パイプラインで変換
# 3 疎行列での変換


# 0 準備 ------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import coo
from sklearn.datasets import load_boston
from sklearn import preprocessing


# データ準備
boston = load_boston()

# 確認
boston.keys()

# データ格納
# --- yは列ベクトルに変換
X, y = boston.data, boston.target.reshape(-1, 1)


# 1 閾値でバイナリ変換 --------------------------------------------------------------------

# データ確認
# --- ラベルデータ
pd.Series(boston.target).hist()
plt.show()

# バイナリ変換
# --- ラベルデータ
# --- 平均値で閾値を設定
new_target = preprocessing.binarize(y, threshold=boston.target.mean())

# データ確認
np.unique(new_target)
np.unique(new_target, return_counts=True)


# バイナリ変換
# --- numpyを使用
new_target_np = (y > y.mean()).astype(int)

# 一致確認
print(all(new_target == new_target_np))


# 2 パイプラインで変換 ---------------------------------------------------------------------

# ＜ポイント＞
# - scikit-learnでは前処理をパイプラインに乗せる
#   --- クラスを用いて一連の処理をパッケージ化
#   --- preprocessing.Binarizer

# インスタンス生成
binar = preprocessing.Binarizer(y.mean())

# バイナリ変換
new_target_pipe = binar.fit_transform(y)

# データ確認
np.unique(new_target_pipe)
np.unique(new_target_pipe, return_counts=True)


# 3 疎行列での変換 -------------------------------------------------------------------------

# ＜疎行列とは＞
# - 0の要素が保存されない行列（メモリ節約）


# 疎行列の作成
spar = coo.coo_matrix(np.random.binomial(1, 0.25, 100))

# バイナリ変換
preprocessing.binarize(spar, threshold=-1)
