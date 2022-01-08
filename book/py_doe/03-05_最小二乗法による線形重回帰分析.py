# ******************************************************************************
# Title     : Pythonで学ぶ実験計画法
# Chapter   : 3 データ解析や回帰分析の手法
# Theme     : 最小二乗法による線形重回帰
# Date      : 2021/11/07
# Page      : P33 - P38
# ******************************************************************************


# ＜概要＞
# - 回帰分析とは説明変数Xで目的変数Yをどれくらい説明できるかを定量的に分析するモデル


# ＜目次＞
# 0 準備
# 1 データ定義
# 2 データの標準化
# 3 メソッドチェーンで記述した場合
# 4 モデル構築
# 5 モデルによる予測
# 6 プロット作成


# 0 準備 ----------------------------------------------------------------

# ライブラリ
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression


# データ準備
df = pd.read_csv('csv/resin.csv', index_col=0, header=0)


# 1 データ定義 ------------------------------------------------------------

# ＜ポイント＞
# - 目的変数はpropertyとして、その他のデータを説明変数とする


# データ定義
# --- 目的変数
# --- 説明変数
y = df.iloc[:, 0]
x = df.iloc[:, 1:]


# 2 データの標準化 --------------------------------------------------------

# ＜ポイント＞
# - 特徴量を全て標準化すると定数項がゼロとなる


# ゼロ・バリアンス・フィルタ
deleting_variables = x.columns[x.std() == 0]
x = x.drop(deleting_variables, axis=1)

# データの標準化
autoscaled_y = (y - y.mean()) / y.std()
autoscaled_x = (x - x.mean()) / x.std()


# 3 メソッドチェーンで記述した場合 -------------------------------------------

# ＜ポイント＞
# - メソッドチェーンで記述すると以下のとおり
#   --- この計算結果は以降では使用しない


# 特徴量の標準化
autoscaled_df = df\
    .drop(df.columns[df.std() == 0], axis=1)\
    .transform(lambda x: (x - x.mean()) / x.std())

# データ定義
autoscaled_y2 = autoscaled_df.iloc[:, 0]
autoscaled_x2 = autoscaled_df.iloc[:, 1:]


# 4 モデル構築 --------------------------------------------------------------

# ＜ポイント＞
# - 線形回帰モデルに標準化したデータを適用すると標準線形回帰モデルとなる
#   --- 目的変数と説明変数の平均が全てゼロとなることで定数項もゼロとなる（重要）


# インスタンス生成
model = LinearRegression()

# モデル構築
model.fit(X=autoscaled_x, y=autoscaled_y)

# データ確認
vars(model)

# 回帰係数をデータフレームとして取得
standard_regression_coefficients = pd.DataFrame(model.coef_)\
    .set_index(x.columns)\
    .set_axis(['standard_regression_coefficients'], axis=1)


# 5 モデルによる予測 -----------------------------------------------------------

# ＜ポイント＞
# - モデル構築に使用したデータの平均値と標準偏差を用いて基準化する（重要）


# 推定用データの定義
# --- 今モデル構築に用いたデータセットの一部を用いる
x_new = df.iloc[:, 1:]

# データ基準化
# --- モデル構築に使用したデータの平均値と標準偏差を用いて基準化する
autoscaled_x_new = (x_new - x.mean()) / x.std()

# 予測
autoscaled_estimated_y_new = model.predict(X=autoscaled_x_new)

# スケール再変換
# --- 基準化したデータを元のスケールに戻す（元データの単位で比較したい場合）
estimated_y_new = autoscaled_estimated_y_new * y.std() + y.mean()
estimated_y_new = pd.DataFrame(estimated_y_new, index=df.index, columns=['estimated_y'])


# 6 プロット作成 -------------------------------------------------------------

# ＜ポイント＞
# - 実測値と推定値を散布図で比較する（実測値のスケールを用いる）
# - 45度の対角線に近いほど推定精度が高い


# ベースプロット
# --- 実測値 vs. 推定値プロット
plt.rcParams['font.size'] = 12
plt.scatter(y, estimated_y_new.iloc[:, 0], c='blue')


# 対角線を作成
y_max = max(y.max(), estimated_y_new.iloc[:, 0].max())
y_min = min(y.min(), estimated_y_new.iloc[:, 0].min())
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')

# 装飾
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('actual y')
plt.ylabel('estimated y')
plt.gca().set_aspect('equal', adjustable='box')

# 表示
plt.show()
