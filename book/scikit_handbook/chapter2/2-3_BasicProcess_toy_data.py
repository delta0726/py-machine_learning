# ******************************************************************************
# Title     : Scikit-Learnデータ分析実装ハンドブック
# Chapter   : 2 Scikit-Learnと開発環境
# Theme     : 2-3 機械学習の基本的な実装
# Created by: Owner
# Created on: 2021/5/13
# Page      : P46 - P49
# ******************************************************************************


# ＜概要＞
# - 機械学習の基本フローの確認
#   --- {sklearn}のサンプルデータのBostonデータセットを使用
#   --- 線形回帰モデルによる回帰問題


# ＜目次＞
# 0 準備
# 1 データ作成
# 2 訓練データと評価データの準備
# 3 アルゴリズムの選択
# 4 学習
# 5 予測


# 0 準備 -------------------------------------------------------------------------------------

# ライブラリ
import pprint

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# データロード
boston = load_boston()

# データ確認
dir(boston)


# 1 データ作成 -----------------------------------------------------------------------------------

# データフレーム作成
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target
df

# 列確認
df.columns

# データ準備
# --- x：説明変数(RM)
# --- y：目的変数(MEDV)
x = df.RM.to_frame()
y = df.MEDV


# 2 訓練データと評価データの準備 ------------------------------------------------------------------

# データ分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# データ確認
# --- Train:Test = 7:3
x_train.shape
y_train.shape
x_test.shape
y_test.shape


# 3 アルゴリズムの選択 ---------------------------------------------------------------------------

# インスタンス作成
# --- 線形回帰モデル
lr = LinearRegression()

# 確認
pprint.pprint(vars(lr))


# 4 学習 ---------------------------------------------------------------------------------------

# 学習
lr.fit(x_train, y_train)

# 確認
pprint.pprint(vars(lr))


# 5 予測 ---------------------------------------------------------------------------------------

# 予測
predict = lr.predict(X=x_test)

# モデル評価
mean_squared_error(y_true=y_test, y_pred=predict)
