# ******************************************************************************
# Title     : Pythonで学ぶ実験計画法
# Chapter   : 3 データ解析や回帰分析の手法
# Theme     : 非線形重回帰分析
# Date      : 2021/11/15
# Page      : P49 - P51
# ******************************************************************************


# ＜概要＞
# - XとYの非線形関係を表現する最も単純な方法は、特徴量に｢二条項｣｢交差項｣を追加すること
#   --- 特徴量が多いと交差項は非常に多くなってしまう点に注意（パターンを絞ったほうが現実的）
#   --- ｢二乗｣が適切かどうかは不明（0.5乗/指数関数/対数関数などが適切な可能性もある）
# - 非線形性を回帰分析で表現するには、事前にXとYの関係性に対する事前知識を持ってモデリングする必要がある
#   --- 機械学習アルゴリズムは事前知識が少なくてもモデリングが可能


# ＜目次＞
# 0 準備
# 1 データ定義
# 2 データ分割
# 3 データの標準化
# 4 モデル構築
# 5 モデルによる予測
# 6 プロット作成
# 7 予測精度の確認
# 8 テストデータによる検証


# 0 準備 ----------------------------------------------------------------

# ライブラリ
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
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
original_x = df.iloc[:, 1:]

# 説明変数の二条項や交差項を追加
x = original_x.copy()
x_square = original_x ** 2

for i in range(original_x.shape[1]):
    for j in range(original_x.shape[1]):
        if i == j:  # 二乗項
            x = pd.concat(
                [x, x_square.rename(columns={x_square.columns[i]: '{0}^2'.format(x_square.columns[i])}).iloc[:, i]],
                axis=1)
        elif i < j:  # 交差項
            x = pd.concat([x, original_x.iloc[:, i] * original_x.iloc[:, j]], axis=1)
            x = x.rename(columns={0: '{0}*{1}'.format(x_square.columns[i], x_square.columns[j])})


# 2 データ分割 ---------------------------------------------------------------

# テストデータのサンプル数
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


# 3 データの標準化 -----------------------------------------------------------

# ゼロ・バリアンス・フィルタ
deleting_variables = x_train.columns[x_train.std() == 0]
x_train = x_train.drop(deleting_variables, axis=1)
x_test = x_test.drop(deleting_variables, axis=1)

# オートスケーリング
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std()
autoscaled_x_train = (x_train - x_train.mean()) / x_train.std()


# 4 モデル構築 ------------------------------------------------------------------

# ＜ポイント＞
# - 特徴量を全て標準化すると定数項がゼロとなる


# インスタンス生成
model = LinearRegression()

# モデル構築
model.fit(X=autoscaled_x_train, y=autoscaled_y_train)

# データ確認
vars(model)

# 回帰係数をデータフレームとして取得
# --- 以降では特に使わない
standard_regression_coefficients = \
    pd.DataFrame(model.coef_,
                 index=x.columns,
                 columns=['standard_regression_coefficients'])


# 5 モデルによる予測 -----------------------------------------------------------

# トレーニングデータの推定
autoscaled_estimated_y_train = model.predict(autoscaled_x_train)

# スケールを元に戻す
estimated_y_train = autoscaled_estimated_y_train * y_train.std() + y_train.mean()

# データフレーム格納
estimated_y_train = pd.DataFrame(estimated_y_train, index=x_train.index,
                                 columns=['estimated_y'])


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
estimated_y_test = pd.DataFrame(estimated_y_test,index=x_test.index)\
        .set_axis(['estimated_y'], axis=1)

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
