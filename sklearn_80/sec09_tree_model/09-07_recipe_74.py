# ******************************************************************************
# Chapter   : 9 決定木アルゴリズムとアンサンブル学習
# Title     : 9-7 ランダムフォレスト回帰を実装する（Recipe74)
# Created by: Owner
# Created on: 2020/12/30
# Page      : P279 - P283
# ******************************************************************************

# ＜概要＞
# - ランダムフォレストは決定木を並列に平均するアンサンブルアルゴリズム


# ＜目次＞
# 0 準備
# 1 層化サンプリング
# 2 モデリング
# 3 モデル評価
# 4 構成要素の決定木にアクセス
# 5 変数重要度


# 0 準備 ------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# データロード
diabetes = load_diabetes()

# データ格納
X = diabetes.data
y = diabetes.target

# 特徴量のラベル
X_feature_names = ['age', 'gender', 'body mass index', 'average blood pressure',
                   'bl_0', 'bl_1', 'bl_2', 'bl_3', 'bl_4', 'bl_5']


# 1 層化サンプリング -----------------------------------------------------------------------------

# 階層サンプリングのため離散化
bins = 50 * np.arange(8)
binned_y = np.digitize(y, bins)

# データ分割
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, stratify=binned_y)


# 2 モデリング -----------------------------------------------------------------------------

# インスタンス生成
rft = RandomForestRegressor()
vars(rft)

# 学習
rft.fit(X_train, y_train)
vars(rft)


# 3 モデル評価 --------------------------------------------------------------------------------

# 予測
y_pred = rft.predict(X_test)

# 平均二乗誤差(MAE)
mean_absolute_error(y_true=y_test, y_pred=y_pred)

# 平均二乗誤差率(MAPE)
(np.abs(y_test - y_pred) / y_test).mean()


# 4 構成要素の決定木にアクセス ------------------------------------------------------------------

# 全体を表示
rft.estimators_

# ツリー数
# --- n_estimators
len(rft.estimators_)


# 5 変数重要度 -------------------------------------------------------------------------------

# 表示
rft.feature_importances_

# プロット表示
fig, ax = plt.subplots(figsize=(10, 5))
bar_rects = ax.bar(np.arange(10), rft.feature_importances_, color='r', align='center')
ax.xaxis.set_ticks(np.arange(10))
ax.set_xticklabels(X_feature_names, rotation='vertical')
plt.show()
