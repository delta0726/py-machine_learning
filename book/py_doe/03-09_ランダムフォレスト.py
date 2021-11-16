# ******************************************************************************
# Title     : Pythonで学ぶ実験計画法
# Chapter   : 3 データ解析や回帰分析の手法
# Theme     : ランダムフォレスト
# Date      : 2021/11/17
# Page      : P56 - P60
# ******************************************************************************


# ＜概要＞
# - ランダムフォレストは多数の決定木を作成して、予測値は全てのツリーの推定値の平均として算出される
#   --- 特徴量とサンプルをランダムに抽出(サブデータセット)して決定木を作成する
#   --- レコードのサンプリングには重複を許容する
#   --- サンプリングで選ばれなかったレコードをOOB(Out-of-Bag)と呼ぶ（予測精度の推定に使用）
# - ランダムフォレストではOOBエラーでハイパーパラメータの選択を行う
#   --- クロスバリデーションを用いる必要がないので計算が効率的になる


# ＜目次＞
# 0 準備
# 1 データ定義
# 2 データ分割
# 3 データ加工
# 4 ハイパーパラメータのチューニング
# 5 モデル構築
# 6 プロット作成
# 7 予測精度の確認
# 8 テストデータによる検証


# 0 準備 ----------------------------------------------------------------

# ライブラリ
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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


# 2 データ分割 ----------------------------------------------------------------

# パラメータ設定
# ---v テストデータのサンプル数
number_of_test_sample = 5

# データ分割
if number_of_test_sample == 0:
    x_train = x.copy()
    x_test = x.copy()
    y_train = y.copy()
    y_test = y.copy()
else:
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=number_of_test_sample,
                                                        shuffle=True, random_state=99)


# 3 データ加工 ---------------------------------------------------------------

# ＜ポイント＞
# - ツリーモデルでは元データをそのまま用いて学習器を生成する（Zスコア変換する必要はない）


# ゼロ・バリアンス・フィルタ
deleting_variables = x_train.columns[x_train.std() == 0]
x_train = x_train.drop(deleting_variables, axis=1)
x_test = x_test.drop(deleting_variables, axis=1)

# レコード数
x_train.shape
x_test.shape


# 4 ハイパーパラメータのチューニング -------------------------------------------

# ＜ポイント＞
# - クロスバリデーションによる木の深さの最適化を行う


# ハイパーパラメータ
# --- テストデータのサンプル数
# --- 決定木における X の数の割合
# --- サブデータセットの数
number_of_test_samples = 5
x_variables_rates = np.arange(1, 11, dtype=float) / 10
number_of_trees = 300

# 格納用リストの定義
# --- 説明変数の数の割合ごとにOOBのr2を格納
r2_oob = []

# チューニング
# --- max_features(mtry)ごとの予測精度を確認
for x_variables_rate in x_variables_rates:
    model = RandomForestRegressor(n_estimators=number_of_trees,
                                  max_features=int(math.ceil(x_train.shape[1] * x_variables_rate)),
                                  oob_score=True)
    model.fit(X=x_train, y=y_train)
    r2_oob.append(r2_score(y_true=y_train, y_pred=model.oob_prediction_))


# プロット
# --- R2が同じ値になるのはサンプル数が少ないことなどに依存
plt.rcParams['font.size'] = 12
plt.scatter(x_variables_rates, r2_oob, c='blue')
plt.xlabel('rate of x-variables')
plt.ylabel('r2 in OOB')
plt.show()

# 最良パラメータ
# --- R2の最大のものを取得
optimal_x_variables_rate = x_variables_rates[np.where(r2_oob == np.max(r2_oob))[0][0]]


# 5 モデル構築 -----------------------------------------------------------------

# ＜ポイント＞
# - 最良パラメータを用いて学習器を生成して予測


# インスタンス生成
# --- max_featuresは最良パラメータを用いて算出
model = RandomForestRegressor(n_estimators=number_of_trees,
                              max_features=int(math.ceil(x_train.shape[1] * optimal_x_variables_rate)),
                              oob_score=True)

# 学習
model.fit(X=x_train, y=y_train)

# 結果確認
vars(model)

# 変数重要度
variable_importance = pd.DataFrame(model.feature_importances_, index=x_train.columns,
                                   columns=['importances'])

# トレーニングデータを用いた予測
estimated_y_train = model.predict(x_train)
estimated_y_train = pd.DataFrame(estimated_y_train, index=x_train.index, columns=['estimated_y'])


# 6 プロット作成 --------------------------------------------------------------

# ＜ポイント＞
# - インサンプルのため完全に予測可能（予測値は45度の対角線上に並ぶ）


# パラメータ設定
plt.rcParams['font.size'] = 12

# プロット定義
# --- 散布図（実測値 vs 推定値）
plt.scatter(y_train, estimated_y_train.iloc[:, 0], c='blue')

# プロット範囲の取得
y_max = max(y_train.max(), estimated_y_train.iloc[:, 0].max())
y_min = min(y_train.min(), estimated_y_train.iloc[:, 0].min())

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


# 7 予測精度の確認 ----------------------------------------------------------

# メトリック出力
# --- R2は0.991
r2_score(y_true=y_train, y_pred=estimated_y_train)
mean_squared_error(y_true=y_train, y_pred=estimated_y_train)
mean_absolute_error(y_true=y_train, y_pred=estimated_y_train)


# 8 テストデータによる検証 --------------------------------------------------

# 予測値の出力
estimated_y_test = model.predict(x_test)
estimated_y_test = pd.DataFrame(estimated_y_test, index=x_test.index, columns=['estimated_y'])

# メトリック出力
# --- R2は0.968
r2_score(y_true=y_test, y_pred=estimated_y_test)
mean_squared_error(y_true=y_test, y_pred=estimated_y_test, squared=False)
mean_absolute_error(y_true=y_test, y_pred=estimated_y_test)

# プロット作成
# --- テストデータなので5サンプルしかない
plt.rcParams['font.size'] = 12
plt.scatter(y_test, estimated_y_test.iloc[:, 0], c='blue')
y_max = max(y_test.max(), estimated_y_test.iloc[:, 0].max())
y_min = min(y_test.min(), estimated_y_test.iloc[:, 0].min())
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('actual y')
plt.ylabel('estimated y')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
