# ******************************************************************************
# Book Name : 機械学習を解釈する技術
# Chapter   : 3 特徴量の重要度を知る
# Theme     : 線形回帰モデルの変数重要度
# Created on: 2021/09/18
# Page      : P54 - P61
# URL       : https://github.com/ghmagazine/ml_interpret_book
# ******************************************************************************


# ＜概要＞
# - シミュレーションデータの作成プロセスとプロットを確認する
# - 線形回帰モデルにおける変数重要度を回帰係数で確認する


# ＜目次＞
# 0 準備
# 1 シミュレーションデータの生成
# 2 プロットによるデータ確認
# 3 線形回帰モデルの特徴量重要度の確認


# 0 準備 -----------------------------------------------------------------------

# ライブラリ
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 自作モジュール
sys.path.append("book/ml_interpret_book")
from mli.visualize import get_visualization_setting

# その他の設定
# --- ランダムシードの設定
# --- pandasの有効桁数設定（小数2桁表示）
# --- Seabornの設定
# --- warningsを非表示に
np.random.seed(42)
pd.options.display.float_format = "{:.2f}".format
sns.set(**get_visualization_setting())
warnings.simplefilter("ignore")


# 1 シミュレーションデータの生成 ----------------------------------------------

# ライブラリ
from sklearn.model_selection import train_test_split


# 関数定義
# --- 乱数でシミュレーションデータの生成してデータ分割したものを出力
def generate_simulation_data(N, beta, mu, Sigma):

    # 多変量正規分布からデータを生成
    X = np.random.multivariate_normal(mu, Sigma, N)

    # ノイズの生成（平均:0 標準偏差:0.1の正規乱数）
    epsilon = np.random.normal(0, 0.1, N)

    # 乱数合成
    y = X @ beta + epsilon

    # 出力
    return train_test_split(X, y, test_size=0.2, random_state=42)


# シミュレーションデータの設定
# --- (1000, 3)の多変量正規乱数を生成
# --- 平均：0 分散：1
N = 1000
J = 3
mu = np.zeros(J)
Sigma = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
beta = np.array([0, 1, 2])

# シミュレーションデータの生成
X_train, X_test, y_train, y_test = \
    generate_simulation_data(N=N, beta=beta, mu=mu, Sigma=Sigma)


# 2 プロットによるデータ確認 --------------------------------------------

# 関数定義
# --- 散布図を3つ並べて表示
def plot_scatters(X, y, var_names):

    # 設定
    # --- Xの列数をプロット取得
    # --- プロットエリアの設定
    J = X.shape[1]
    fig, axes = plt.subplots(nrows=1, ncols=J, figsize=(4 * J, 4))

    # プロット作成
    for d, ax in enumerate(axes):
        sns.scatterplot(X[:, d], y, alpha=0.3, ax=ax)
        ax.set(xlabel=var_names[d],
               ylabel="Y",
               xlim=(X.min() * 1.1, X.max() * 1.1)
               )
        fig.show()


# 可視化
var_names = [f"X{j}" for j in range(J)]
plot_scatters(X=X_train, y=y_train, var_names=var_names)


# 3 線形回帰モデルの特徴量重要度の確認 --------------------------------------

# ライブラリ
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# 関数定義
# --- 回帰係数の大きさを確認する棒グラフを作成
def plot_bar(variables, values, xlabel=None, ylabel=None, title=None):
    fig, ax = plt.subplots()
    ax.barh(variables, values)
    ax.set(xlabel=xlabel, ylabel=ylabel, ylim=(0, None))
    fig.suptitle(title)
    fig.show()


# 学習
# --- 線形回帰モデル
lm = LinearRegression()
lm.fit(X_train, y_train)

# 結果確認
vars(lm)

# 予測精度の確認
# --- R2=0.997
r2_score(y_true=y_test, y_pred=lm.predict(X_test))

# 変数重要度の確認
# --- 回帰係数のプロット作成
plot_bar(variables=var_names, values=lm.coef_, xlabel="Importance",
         title="Permutation Importance")
