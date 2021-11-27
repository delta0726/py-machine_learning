# ******************************************************************************
# Title     : Pythonで学ぶ実験計画法
# Chapter   : 4 モデルの適用範囲
# Theme     : アンサンブル学習
# Date      : 2021/11/27
# Page      : P98 - P102
# ******************************************************************************


# ＜概要＞
# - アンサンブル学習法では予測値ごとの標準偏差の大きさからADを設定する
#   --- 標準偏差が大きいほど信頼性が低いことを意味するためADの判断材料となる
#   --- ここではアンサンブル学習は複数モデルを組み合わせること（ランダムフォレストではない）


# ＜目次＞
# 0 準備
# 1 データ定義
# 2 データ分割
# 3 アンサンブル学習
# 4 予測値の取得
# 5 ADの確認


# 0 準備 ----------------------------------------------------------------

# ライブラリ
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# データ準備
df = pd.read_csv('csv/resin.csv', index_col=0, header=0)
x_pred = pd.read_csv('csv/resin_prediction.csv', index_col=0, header=0)


# 1 データ定義 ------------------------------------------------------------

# ＜ポイント＞
# - 目的変数はpropertyとして、その他のデータを説明変数とする


# データ定義
# --- 目的変数
# --- 説明変数
y = df.iloc[:, 0]
x = df.iloc[:, 1:]


# 2 データ加工 ---------------------------------------------------------------

# ゼロ・バリアンス・フィルタ
deleting_variables = x.columns[x.std() == 0]
x = x.drop(deleting_variables, axis=1)
x_pred = x_pred.drop(deleting_variables, axis=1)

# データの標準化
autoscaled_x = (x - x.mean()) / x.std()
autoscaled_x_pred = (x_pred - x.mean()) / x.std()
autoscaled_y = (y - y.mean()) / y.std(ddof=1)


# 3 アンサンブル学習 -----------------------------------------------------------

# ＜ポイント＞
# - 特徴量をランダムに選択することで複数モデルを構築
# - ガウスカーネルSVRによりモデリング（ハイパーパラメータのチューニングも行う）

# パラメータ設定
# --- サブデータセットの数
# --- 各サブデータセットで選択される説明変数の数の割合
# --- N-fold CV の N
number_of_sub_datasets = 30
rate_of_selected_x_variables = 0.75
fold_number = 10

# ハイパーパラメータの設定（ガウスカーネルSVR）
# --- C の候補
# --- ε の候補
# --- γ の候補
svr_cs = 2 ** np.arange(-5, 11, dtype=float)
svr_epsilons = 2 ** np.arange(-10, 1, dtype=float)
svr_gammas = 2 ** np.arange(-20, 11, dtype=float)

# 特徴量の抽出数
number_of_x_variables = int(np.ceil(x.shape[1] * rate_of_selected_x_variables))
print('各サブデータセットにおける説明変数の数 :', number_of_x_variables)

# 格納用のオブジェクト作成
# --- 各サブデータセットの説明変数の番号を追加
# --- SVRモデル
selected_x_variable_numbers = []
submodels = []


submodel_number = 0
for submodel_number in range(number_of_sub_datasets):

    # 進捗状況の表示
    print(submodel_number + 1, '/', number_of_sub_datasets)

    # 説明変数の選択
    # --- 0 から 1 までの間に一様乱数を説明変数の数だけ生成
    # --- その乱数値が小さい順に説明変数を5個のうちn個を選択
    random_x_variables = np.random.rand(x.shape[1])
    selected_x_variable_numbers_tmp = random_x_variables.argsort()[:number_of_x_variables]
    selected_autoscaled_x = autoscaled_x.iloc[:, selected_x_variable_numbers_tmp]
    selected_x_variable_numbers.append(selected_x_variable_numbers_tmp)

    # ハイパーパラメータの最適化
    # --- 分散最大化によるガウシアンカーネルのγの最適化
    variance_of_gram_matrix = []
    selected_autoscaled_x_array = np.array(selected_autoscaled_x)

    for nonlinear_svr_gamma in svr_gammas:
        gram_matrix = np.exp(- nonlinear_svr_gamma * ((selected_autoscaled_x_array[:, np.newaxis] - selected_autoscaled_x_array) ** 2).sum(axis=2))
        variance_of_gram_matrix.append(gram_matrix.var(ddof=1))

    optimal_svr_gamma = svr_gammas[np.where(variance_of_gram_matrix==np.max(variance_of_gram_matrix))[0][0]]

    # CVによる最適化
    # --- ε
    model_in_cv = GridSearchCV(SVR(kernel='rbf', C=3, gamma=optimal_svr_gamma),
                               {'epsilon': svr_epsilons}, cv=fold_number)
    model_in_cv.fit(selected_autoscaled_x, autoscaled_y)
    optimal_svr_epsilon = model_in_cv.best_params_['epsilon']

    # CVによる最適化
    # --- C
    model_in_cv = GridSearchCV(SVR(kernel='rbf', epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma),
                               {'C': svr_cs}, cv=fold_number)
    model_in_cv.fit(selected_autoscaled_x, autoscaled_y)
    optimal_svr_c = model_in_cv.best_params_['C']

    # CVによる最適化
    # --- γ
    model_in_cv = GridSearchCV(SVR(kernel='rbf', epsilon=optimal_svr_epsilon, C=optimal_svr_c),
                               {'gamma': svr_gammas}, cv=fold_number)
    model_in_cv.fit(selected_autoscaled_x, autoscaled_y)
    optimal_svr_gamma = model_in_cv.best_params_['gamma']

    # モデルの構築
    # --- SVR
    submodel = SVR(kernel='rbf', C=optimal_svr_c, epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma)  # モデルの宣言
    submodel.fit(selected_autoscaled_x, autoscaled_y)
    submodels.append(submodel)


# 4 予測値の取得 -------------------------------------------------------------------

# ＜ポイント＞
# - サブモデルごとの予測値を取得してデータフレームに列方向に格納する


# 空のオブジェクト準備
# --- サブモデルごとの予測用データセットの y の推定結果を追加
estimated_y_prediction_all = pd.DataFrame()

# 予測用データセットの y の推定
submodel_number = 0
for submodel_number in range(number_of_sub_datasets):
    # 説明変数の選択
    selected_autoscaled_x_pred = \
        autoscaled_x_pred.iloc[:, selected_x_variable_numbers[submodel_number]]

    # 予測値の取得
    # --- 格納したサブモデルから予測用データセットで予測値を推定
    estimated_y_prediction = \
        pd.DataFrame(submodels[submodel_number].predict(selected_autoscaled_x_pred))

    # スケール変換
    # --- 元のスケールに戻す
    estimated_y_prediction = estimated_y_prediction * y.std() + y.mean()

    # データフレーム格納
    estimated_y_prediction_all = \
        pd.concat([estimated_y_prediction_all, estimated_y_prediction], axis=1)


# 結果確認
estimated_y_prediction_all


# 5 ADの確認 -------------------------------------------------------------------

# ＜ポイント＞
# - 複数モデルの予測値の標準偏差をもとにADを設定する（サンプルごとの標準偏差の水準で評価）
# - 分類問題の場合は予測確率やLogLossを用いて連続値で評価する方が適切なようだ（書籍ではクラス割合）


# 推定値の平均値
estimated_y_prediction = pd.DataFrame(estimated_y_prediction_all.mean(axis=1))
estimated_y_prediction.index = x_pred.index
estimated_y_prediction.columns = ['estimated_y']

# 予測用データセットの推定値の標準偏差
std_of_estimated_y_prediction = pd.DataFrame(estimated_y_prediction_all.std(axis=1))
std_of_estimated_y_prediction.index = x_pred.index
std_of_estimated_y_prediction.columns = ['std_of_estimated_y']

# 確認
estimated_y_prediction
std_of_estimated_y_prediction
