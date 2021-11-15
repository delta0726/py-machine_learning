# ******************************************************************************
# Title     : Pythonで学ぶ実験計画法
# Chapter   : 3 データ解析や回帰分析の手法
# Theme     : 決定木
# Date      : 2021/11/15
# Page      : P52 - P55
# ******************************************************************************


# ＜概要＞
# - 決定木はXの特徴量空間をYが類似したサンプルのみが存在する領域を設定して分割する
# - 新しいサンプルの予測値は該当する領域におけるYの平均値として決定される
#   --- 同じ値の予測値が出力されることがある


# ＜目次＞
# 0 準備
# 1 データ定義
# 2 データ分割
# 3 データ加工
# 4 ハイパーパラメータのチューニング
# 5 モデル構築
# 6 モデルによる予測
# 7 プロット作成
# 8 予測精度の確認
# 9 テストデータによる検証


# 0 準備 ----------------------------------------------------------------

# ライブラリ
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
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


# 4 ハイパーパラメータのチューニング -------------------------------------------

# ＜ポイント＞
# - クロスバリデーションによる木の深さの最適化を行う


# パラメータ設定
# --- クロスバリデーションのfold数
# --- 木の深さの最大値の候補（学習データのサンプル数から考えて4以上は無駄）
# --- 葉ノードごとのサンプル数の最小値
fold_number = 10
max_depths = np.arange(1, 31)
min_samples_leaf = 3

# インスタンス生成
cross_validation = KFold(n_splits=fold_number, random_state=9, shuffle=True)

# 格納用リストの定義
r2cvs = []

# チューニング
# --- 訓練データを用いる
for max_depth in max_depths:
    model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=59)
    estimated_y_in_cv = cross_val_predict(estimator=model, X=x_train, y=y_train, cv=cross_validation)
    r2cvs.append(r2_score(y_train, estimated_y_in_cv))

# プロット
# --- R2が同じ値になるのはサンプル数が少ないことなどに依存
plt.rcParams['font.size'] = 12
plt.scatter(max_depths, r2cvs, c='blue')
plt.xlabel('maximum depth of tree')
plt.ylabel('r^2 in cross-validation')
plt.show()

# 最良パラメータ
# --- R2の最大のものを取得
optimal_max_depth = max_depths[np.where(r2cvs == np.max(r2cvs))[0][0]]


# 5 モデル構築 -----------------------------------------------------------------

# ＜ポイント＞
# - 最良パラメータを用いて学習器を生成して予測


# インスタンス生成
# --- max_depthは最良パラメータを使用
model = DecisionTreeRegressor(max_depth=optimal_max_depth, min_samples_leaf=min_samples_leaf,
                              random_state=59)

# 学習
model.fit(X=x_train, y=y_train)

# 結果確認
vars(model)


# 6 モデルによる予測 -----------------------------------------------------------

# トレーニングデータの推定
estimated_y_train = model.predict(x_train)
estimated_y_train = pd.DataFrame(estimated_y_train, index=x_train.index, columns=['estimated_y'])

# データフレーム格納
y_train_for_save = pd.DataFrame(y_train).set_axis(['actual_y'], axis=1)
y_error_train = y_train_for_save.iloc[:, 0] - estimated_y_train.iloc[:, 0]
y_error_train = pd.DataFrame(y_error_train).set_axis(['error_of_y'], axis=1)
results_train = pd.concat([y_train_for_save, estimated_y_train, y_error_train], axis=1)


# 7 プロット作成 --------------------------------------------------------------

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


# 8 予測精度の確認 ----------------------------------------------------------

# メトリック出力
# --- R2は1、MSEとMAEは0
r2_score(y_true=y_train, y_pred=estimated_y_train)
mean_squared_error(y_true=y_train, y_pred=estimated_y_train)
mean_absolute_error(y_true=y_train, y_pred=estimated_y_train)


# 9 テストデータによる検証 --------------------------------------------------

# 予測値の出力
estimated_y_test = model.predict(x_test)
estimated_y_test = pd.DataFrame(estimated_y_test, index=x_test.index, columns=['estimated_y'])

# メトリック出力
# --- R2は0.991
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
