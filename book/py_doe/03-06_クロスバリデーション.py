# ******************************************************************************
# Title     : Pythonで学ぶ実験計画法
# Chapter   : 3 データ解析や回帰分析の手法
# Theme     : 回帰モデルの推定精度の評価（クロスバリデーション）
# Date      : 2021/11/09
# Page      : P38 - P49
# ******************************************************************************


# ＜概要＞
# - モデルの推定精度の評価方法としてクロスバリデーションを確認する
#   --- クロスバリデーションは全てのデータをテストデータとして推定してモデル評価を行う
#   --- データのサンプル数が少ない場合に有効（計算量はK倍になる）


# ＜目次＞
# 0 準備
# 1 データ定義
# 2 データの標準化
# 3 モデル構築
# 4 モデルによる予測
# 5 プロット作成
# 6 予測精度の確認
# 7 クロスバリデーションによる予測値の推定


# 0 準備 ----------------------------------------------------------------

# ライブラリ
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


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


# 3 モデル構築 --------------------------------------------------------------

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
# --- 以降では特に使わない
standard_regression_coefficients = pd.DataFrame(model.coef_)\
    .set_index(x.columns)\
    .set_axis(['standard_regression_coefficients'], axis=1)


# 4 モデルによる予測 -----------------------------------------------------------

# ＜ポイント＞
# - 訓練データで作成したモデルで訓練データを予測（インサンプル）


# トレーニングデータの推定
autoscaled_estimated_y = model.predict(autoscaled_x)

# スケールを元に戻す
estimated_y = autoscaled_estimated_y * y.std() + y.mean()

# データフレーム格納
estimated_y = pd.DataFrame(estimated_y, index=x.index)\
        .set_axis(['estimated_y'], axis=1)


# 5 プロット作成 --------------------------------------------------------------

# ＜ポイント＞
# - インサンプルのため完全に予測可能（予測値は45度の対角線上に並ぶ）


# パラメータ設定
plt.rcParams['font.size'] = 12

# プロット定義
# --- 散布図（実測値 vs 推定値）
plt.scatter(y, estimated_y.iloc[:, 0], c='blue')

# プロット範囲の取得
y_max = max(y.max(), estimated_y.iloc[:, 0].max())
y_min = min(y.min(), estimated_y.iloc[:, 0].min())

# プロット設定
# --- 取得した最小値-5%から最大値+5%まで、対角線を作成
# --- 図の形を正方形にする
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('actual y')
plt.ylabel('estimated y')
plt.gca().set_aspect('equal', adjustable='box')

# プロット表示
plt.show()


# 6 予測精度の確認 ----------------------------------------------------------

# ＜ポイント＞
# - インサンプルの予測精度の確認

# メトリック出力
# --- R2は1、MSEとMAEは0
r2_score(y_true=y, y_pred=estimated_y)
mean_squared_error(y_true=y, y_pred=estimated_y, squared=False)
mean_absolute_error(y_true=y, y_pred=estimated_y)


# 7 クロスバリデーションによる予測値の推定 --------------------------------------

# ＜ポイント＞
# - モデルデータを10Foldに分割して9つを用いたモデルで残り1つを予測する
#   --- イテレーションで10回繰り返すことでデータセット全ての予測値が得られる
#   --- 外部バリデーションの場合はテストデータ(全体のX％)のみ予測値が得られる


# インスタンス生成
# --- クロスバリデーションの分割の設定（10Fold）
cross_validation = KFold(n_splits=10, random_state=9, shuffle=True)

# 予測値の推定
autoscaled_estimated_y_in_cv = \
    cross_val_predict(estimator=model, X=autoscaled_x, y=autoscaled_y,
                      cv=cross_validation)

# 元のスケールに戻す
estimated_y_in_cv = autoscaled_estimated_y_in_cv * y.std() + y.mean()

# メトリック出力
# --- R2は0.859
r2_score(y_true=y, y_pred=estimated_y_in_cv)
mean_squared_error(y_true=y, y_pred=estimated_y_in_cv)
mean_absolute_error(y_true=y, y_pred=estimated_y_in_cv)

# プロット作成
# --- 全てのデータで予測値が出ているので20サンプル表示される
plt.rcParams['font.size'] = 12
plt.scatter(y, estimated_y_in_cv.iloc[:, 0], c='blue')
y_max = max(y.max(), estimated_y_in_cv.iloc[:, 0].max())
y_min = min(y.min(), estimated_y_in_cv.iloc[:, 0].min())
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('actual y')
plt.ylabel('estimated y')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
