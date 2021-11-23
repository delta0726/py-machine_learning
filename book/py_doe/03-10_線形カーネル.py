# ******************************************************************************
# Title     : Pythonで学ぶ実験計画法
# Chapter   : 3 データ解析や回帰分析の手法
# Theme     : サポートベクター回帰（線形カーネル）
# Date      : 2021/11/18
# Page      : P60 - P67
# ******************************************************************************


# ＜概要＞
# - クラス分類手法であるサポートベクターマシンを回帰分析に応用した手法
#   --- カーネルトリックにより線形モデルを非線形に拡張している


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
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.svm import SVR
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
# --- テストデータのサンプル数
number_of_test_samples = 5

# データ分割
if number_of_test_samples == 0:
    x_train = x.copy()
    x_test = x.copy()
    y_train = y.copy()
    y_test = y.copy()
else:
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=number_of_test_samples,
                                                        shuffle=True,
                                                        random_state=99)


# 3 データ加工 ---------------------------------------------------------------

# ＜ポイント＞
# - ツリーモデルでは元データをそのまま用いて学習器を生成する（Zスコア変換する必要はない）


# ゼロ・バリアンス・フィルタ
deleting_variables = x_train.columns[x_train.std() == 0]
x_train = x_train.drop(deleting_variables, axis=1)
x_test = x_test.drop(deleting_variables, axis=1)

# データの標準化
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std()
autoscaled_x_train = (x_train - x_train.mean()) / x_train.std()


# 4 ハイパーパラメータのチューニング ----------------------------------------------

# パラメータ設定
# --- クロスバリデーションのFold数
# --- 線形SVR のCの候補
# --- 線形SVRのεの候補
fold_number = 10
linear_svr_cs = 2 ** np.arange(-10, 5, dtype=float)
linear_svr_epsilons = 2 ** np.arange(-10, 0, dtype=float)

# クロスバリデーションの分割の設定
cross_validation = KFold(n_splits=fold_number, random_state=9, shuffle=True)

# グリッドサーチの設定
gs_cv = GridSearchCV(SVR(kernel='linear'),
                     {'C': linear_svr_cs, 'epsilon': linear_svr_epsilons},
                     cv=cross_validation)

# グリッドサーチ + クロスバリデーション実施
gs_cv.fit(autoscaled_x_train, autoscaled_y_train)

# 最適パラメータ
# --- C
# --- ε
optimal_linear_svr_c = gs_cv.best_params_['C']
optimal_linear_svr_epsilon = gs_cv.best_params_['epsilon']


# 5 モデル構築 -----------------------------------------------------------------------

# インスタンス生成
model = SVR(kernel='linear', C=optimal_linear_svr_c, epsilon=optimal_linear_svr_epsilon)

# モデル構築
model.fit(X=autoscaled_x_train, y=autoscaled_y_train)

# 確認
vars(model)

# 回帰係数をデータフレームとして取得
# --- 以降では特に使わない
standard_regression_coefficients = \
    pd.DataFrame(model.coef_.T,
                 index=x_train.columns,
                 columns=['standard_regression_coefficients'])

# トレーニングデータを用いた予測
autoscaled_estimated_y_train = model.predict(autoscaled_x_train)

# スケールをもとに戻す
estimated_y_train = autoscaled_estimated_y_train * y_train.std() + y_train.mean()
estimated_y_train = pd.DataFrame(estimated_y_train, index=x_train.index, columns=['estimated_y'])


# 6 プロット作成 ----------------------------------------------------------------------

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
# --- R2は1、MSEとMAEは0
r2_score(y_true=y_train, y_pred=estimated_y_train)
mean_squared_error(y_true=y_train, y_pred=estimated_y_train)
mean_absolute_error(y_true=y_train, y_pred=estimated_y_train)


# 8 テストデータによる検証 --------------------------------------------------

# データ標準化
# --- 訓練データに対して基準化する点に注意
autoscaled_x_test = (x_test - x_train.mean()) / x_train.std()

# 予測値の出力
autoscaled_estimated_y_test = model.predict(autoscaled_x_test)

# スケールを元に戻す
estimated_y_test = autoscaled_estimated_y_test * y_train.std() + y_train.mean()

# データフレーム格納
estimated_y_test = pd.DataFrame(estimated_y_test, index=x_test.index, columns=['estimated_y'])

# メトリック出力
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
