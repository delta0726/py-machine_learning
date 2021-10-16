# ******************************************************************************
# Book Name : 機械学習を解釈する技術
# Chapter   : 2 線形回帰モデルを通して｢解釈性｣を理解する
# Created on: 2021/8/15
# Page      : P29 - P51
# URL       : https://github.com/ghmagazine/ml_interpret_book
# ******************************************************************************


# ＜目次＞
# 0 準備
# 1 データ準備
# 2 データ前処理
# 3 データ分割
# 4 線形モデルの学習と評価
# 5 モデル精度の計算
# 6 線形回帰モデルの解釈
#   6-1 特徴量と予測値の平均的な関係
#   6-2 インスタンスごとの特徴量と予測値の関係
#   6-3 特徴量の重要度
#   6-4 予測の理由
# 7 ランダムフォレストによる予測


# 0 準備 --------------------------------------------------------------------

# ライブラリ
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# from __future__ import annotations

# 自作モジュール
from mli.visualize import get_visualization_setting

# その他設定
np.random.seed(42)
pd.options.display.float_format = "{:.2f}".format
sns.set(**get_visualization_setting())
warnings.simplefilter("ignore")

# 1 データ準備 -----------------------------------------------------------------

from sklearn.datasets import load_boston

# データセットのロード
boston = load_boston()

# データ格納
X = pd.DataFrame(data=boston["data"], columns=boston["feature_names"])
y = boston["target"]

# データ確認
# --- 特徴量の意味はP33で確認
X.head()
y[0:4]


# 2 データ前処理 ----------------------------------------------------------------

# ＜ポイント＞
# - 住宅価格は右に裾の長いヒストグラムになっている
#   --- 解釈性を重視してyの対数変換は行わない

# 関数定義
# --- ヒストグラム作成
def plot_histogram(x, title=None, x_label=None):
    fig, ax = plt.subplots()
    sns.distplot(x, kde=False, ax=ax)
    fig.suptitle(title)
    ax.set_xlabel(x_label)
    fig.show()


# プロット作成
# --- ラベルデータ（住宅価格）
plot_histogram(x=y, title="目的変数の分布", x_label="NEDV")


# 関数定義
# --- 散布図の作成
# --- yとの関係をプロット（変数間の散布図ではない）
def plot_scatters(X, y, title=None):
    cols = X.columns
    fig, axes = plt.subplots(nrows=2, ncols=2)

    for ax, c in zip(axes.ravel(), cols):
        sns.scatterplot(X[c], y, ci=None, ax=ax)
        ax.set(ylabel="MEDV")

    fig.suptitle(title)
    fig.show()


# プロット作成
plot_scatters(X=X[["RM", "LSTAT", "DIS", "CRIM"]],
              y=y,
              title="目的変数と各特徴量の関係")

# 3 データ分割 -----------------------------------------------------------------

from sklearn.model_selection import train_test_split

# データ分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4 線形モデルの学習と評価 ----------------------------------------------------------

from sklearn.linear_model import LinearRegression

# モデル構築＆学習
# --- インスタンス生成
# --- 学習
lm = LinearRegression()
lm.fit(X_train, y_train)

# 確認
vars(lm)


# 5 モデル精度の計算 ---------------------------------------------------------------

# ＜ポイント＞
# - 線形回帰モデルはMSEが最小となるように回帰係数を決定する
#   --- RMSEは直接的に最小化しているMSEを集計したもの

from sklearn.metrics import mean_squared_error, r2_score


# 関数定義
# --- 学習器からメトリックを出力
def regressor_metrics(estimator, X, y):
    y_pred = estimator.predict(X)
    df = pd.DataFrame(
        data={
            "RMSE": [mean_squared_error(y_true=y, y_pred=y_pred, squared=False)],
            "R2": [r2_score(y_true=y, y_pred=y_pred)]
        }
    )
    return df


# 精度評価
# - RMSEは誤差指標（小さいほうが良好）
# - R2は正解率指標（大きいほうが良好）
regressor_metrics(estimator=lm, X=X_test, y=y_test)


# 6 線形回帰モデルの解釈 ------------------------------------------------------------

# 6-1 特徴量と予測値の平均的な関係 --------------------------

# ＜ポイント＞
# --- 線形回帰モデルの回帰係数は特徴量と予測値の"平均的な関係"を示す
# --- 回帰係数は各特徴量が1単位だけ大きくなった時のyの変化を示す

# 関数定義
# --- 切片と回帰係数の取得
def get_coef(estimator, var_names):
    df = pd.DataFrame(
        data={"coef": [estimator.intercept_] + estimator.coef_.tolist()},
        index=["intercept"] + var_names
    )
    return df


# 回帰係数の取り出し
df_coef = get_coef(estimator=lm, var_names=X.columns.tolist())
df_coef.T


# 6-2 インスタンスごとの特徴量と予測値の関係 ------------------

# ＜ポイント＞
# - インスタンスごとの特徴量のインパクトは、インスタンスの当該インプット値に応じて異なる
#   --- 回帰係数は全インスタンスで同じなのでインパクトの方向性は同じ（インパクト量の違い）
#   --- インパクトは微分により求めることができる（計算を解りやすくするため二乗項を追加）


# コピー
# --- 元データの上書きを回避
X_train2 = X_train.copy()
X_test2 = X_test.copy()

# データ加工
# --- コピーデータに二乗項を追加
X_train2["LSTAT2"] = X_train2["LSTAT"] ** 2
X_test2["LSTAT2"] = X_test2["LSTAT"] ** 2

# データ確認
X_train2
X_test2

# モデル構築＆学習
# --- インスタンス生成
# --- 学習
lm2 = LinearRegression()
lm2.fit(X_train2, y_train)

# 予測精度の評価
regressor_metrics(estimator=lm2, X=X_test2, y=y_test)

# 予測精度の評価
df_coef2 = get_coef(estimator=lm2, var_names=X_train2.columns.tolist())
df_coef2.T


# 関数定義
# --- 元のフォーミュラを微分して変化率(インパクト)を算出
def calc_lstat_impact(df, lstat):
    return (df.loc["LSTAT"] + 2 * df.loc["LSTAT2"] * lstat).values[0]


# インパクト出力
i = 274
lstat = X_test2.loc[i, "LSTAT"]
impact = calc_lstat_impact(df_coef2, lstat)
print(f"インスタンス{i}でLSTATが1単位増加したときの効果(LSTAT={lstat:.2f})：{impact:.2f}")

i = 491
lstat = X_test2.loc[i, "LSTAT"]
impact = calc_lstat_impact(df_coef2, lstat)
print(f"インスタンス{i}でLSTATが1単位増加したときの効果(LSTAT={lstat:.2f})：{impact:.2f}")


# 6-3 特徴量の重要度 ---------------------------------------

# ＜ポイント＞
# - 線形回帰モデルでは回帰係数が1単位の変化量(重要度)を示している
# - データを基準化しないと1単位の変化量が特徴量ごとに異なって判断できない


# 基準化なしの回帰係数の評価 -------------------------

# 回帰係数の確認
# --- NOXの回帰係数が-17.20と絶対値が最も大きい
# --- しかし、データを基準化しないで回帰モデルを実行したためNOXが最も重要かどうか判断できない
df_coef

# 特徴量のレンジを確認
# --- NOXのレンジは0.49と1以下なので、回帰係数は割り引いて考える必要がある
df_range = pd.DataFrame(data={"range": X_train.max() - X_train.min()})
df_range

# 基準化ありの回帰係数の評価 -------------------------

from sklearn.preprocessing import StandardScaler

# インスタンス生成＆学習
# --- 標準化に関係する要素が生成される
ss = StandardScaler()
ss.fit(X_train)
vars(ss)

# 標準化
# --- 標準化したデータに変換
X_train_ss = ss.fit_transform(X_train)
X_test_ss = ss.fit_transform(X_test)

# 学習
# --- 訓練データ
lm_ss = LinearRegression()
lm_ss.fit(X_train_ss, y_train)
vars(lm_ss)

# 予測精度の評価
# --- 検証データ
regressor_metrics(lm_ss, X_test_ss, y_test)

# 回帰係数
df_coef_ss = get_coef(estimator=lm_ss, var_names=X_train.columns.tolist())
df_coef_ss


# 6-4 予測の理由 -------------------------------------------

# 先頭のインスタンスを抽出
Xi = X_test.iloc[[0]]

# 回帰係数の取得
df_coef

# 寄与度の算出
# --- 入力値 * 回帰係数
Xi.T * df_coef.drop("intercept").values


# 7 ランダムフォレストによる予測 --------------------------------------------

# ＜ポイント＞
# - ランダムフォレストはパラメータチューニングをしなくても予測精度が高い
# - ブラックボックスモデルで線形回帰のような解釈ができない
# --- 次章以降の手法を用いてモデル解釈を行う

from sklearn.ensemble import RandomForestRegressor

# インスタンスの生成
# --- ランダムフォレスト回帰
# --- n_jobsを-1に設定することで全てのCPUを使用して並列化
rf = RandomForestRegressor(n_jobs=-1, random_state=42)
vars(rf)

# 学習
# --- 訓練データ
rf.fit(X_train, y_train)
vars(rf)

# 予測精度の評価
# --- テストデータ
#   --- R2でみるとlmは0.67であるのに対してrfは0.89と高い
regressor_metrics(estimator=rf, X=X_test, y=y_test)
