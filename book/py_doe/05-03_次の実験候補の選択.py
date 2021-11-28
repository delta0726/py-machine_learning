# ******************************************************************************
# Title     : Pythonで学ぶ実験計画法
# Chapter   : 5 実験計画法・適応的実験計画法の実践
# Theme     : 次の実験候補の選択
# Date      : 2021/11/28
# Page      : P107 - P116
# ******************************************************************************


# ＜概要＞
# - 予測値が目標値となるようなシミュレーション設定を把握する


# ＜目次＞
# 0 準備
# 1 シミュレーション設定
# 2 データ定義
# 3 データ非線形変換
# 4 データ加工
# 5 カーネルの設定
# 6 モデル構築
# 7 モデル学習
# 8 予測値の取得
# 9 プロット作成
# 10 クロスバリデーション予測
# 11 予測データの取得
# 12 ADによる判定


# 0 準備 -----------------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, OneClassSVM
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Matern, DotProduct
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import NearestNeighbors


# データロード
# --- 本来のデータセット（dfはどちらかを選択）
# --- シミュレーションデータセットのうちD最適化基準で選択されたもの（dfはどちらかを選択）
# --- シミュレーションデータセットのうちD最適化基準で選択されたもの以外
df = pd.read_csv('csv/resin.csv', index_col=0, header=0)
# df = pd.read_csv('csv/resin.csv', index_col=0, header=0)
x_pred = pd.read_csv('csv/remaining_samples.csv', index_col=0, header=0)


# 1 シミュレーション設定 --------------------------------------------------

# ＜モデルのパターン＞
# ols_linear     : 線形回帰(OLS)
# ols_nonlinear  : 非線形重回帰
# svr_linear     : 線形カーネルを用いたサポートベクター回帰
# svr_gaussian   : ガウシアンカーネルを用いたサポートベクター回帰
# gpr_one_kernel : カーネル関数を1つ選択したガウス過程回帰
# gpr_kernels    : カーネル関数を1つ選択したガウス過程回帰（クロスバリデーション）

# ＜AD法のパターン＞
# knn                      : k近傍法
# ocsvm                    : サポートベクターマシン（One-Class）
# ocsvm_gamma_optimization : サポートベクターマシン（ガンマ最適化）

# パラメータ設定
# --- 回帰分析手法
# --- AD設定手法
regression_method = 'ols_nonlinear'
ad_method = 'ocsvm'


# 2 データ定義 ------------------------------------------------------------

# ＜ポイント＞
# - 目的変数はpropertyとして、その他のデータを説明変数とする


# データ定義
# --- 目的変数
# --- 説明変数
y = df.iloc[:, 0]
x = df.iloc[:, 1:]


# 3 データ非線形変換 --------------------------------------------------------

# ＜ポイント＞
# - 二乗項と交差項をループ処理により追加
#   --- 二重ループでi=jなら二乗項、i!=jなら交差項を追加していく（元の特徴量を掛け算）
#   --- 必ずしもこの非線形変換が適切でない点に注意


# 学習データの特徴量の拡張
# --- 非線形変換で二乗項と交差項を追加
i = 0
j = 0
if regression_method == 'ols_nonlinear':
    x_tmp = x.copy()
    x_pred_tmp = x_pred.copy()
    # 二乗項の作成
    x_square = x ** 2
    x_pred_square = x_pred ** 2
    # 追加
    for i in range(x_tmp.shape[1]):
        for j in range(x_tmp.shape[1]):
            # 二乗項
            if i == j:
                item_tra = {x_square.columns[i]: '{0}^2'.format(x_square.columns[i])}
                item_prd = {x_pred_square.columns[i]: '{0}^2'.format(x_pred_square.columns[i])}
                x = pd.concat([x, x_square.rename(columns=item_tra).iloc[:, i]], axis=1)
                x_pred = pd.concat([x_pred, x_pred_square.rename(columns=item_prd).iloc[:, i]], axis=1)
            # 交差項
            elif i < j:
                x_cross = x_tmp.iloc[:, i] * x_tmp.iloc[:, j]
                x_pred_cross = x_pred_tmp.iloc[:, i] * x_pred_tmp.iloc[:, j]
                x_cross.name = '{0}*{1}'.format(x_tmp.columns[i], x_tmp.columns[j])
                x_pred_cross.name = '{0}*{1}'.format(x_pred_tmp.columns[i], x_pred_tmp.columns[j])
                x = pd.concat([x, x_cross], axis=1)
                x_pred = pd.concat([x_pred, x_pred_cross], axis=1)


# データ確認
pd.DataFrame(x.columns)
pd.DataFrame(x_pred.columns)


# 4 データ加工 ---------------------------------------------------------------

# ＜ポイント＞
# - ツリーモデルではないのでZスコアにスケーリングしておく


# ゼロバリアンス・フィルタ
deleting_variables = x.columns[x.std() == 0]
x = x.drop(deleting_variables, axis=1)
x_pred = x_pred.drop(deleting_variables, axis=1)

# 列名の設定
x_pred.columns = x.columns

# オートスケーリング
autoscaled_y = (y - y.mean()) / y.std()
autoscaled_x = (x - x.mean()) / x.std()
autoscaled_x_pred = (x_pred - x.mean()) / x.std()

# データ確認
pd.concat([autoscaled_y, autoscaled_x], axis=1)
autoscaled_x_pred


# 5 カーネルの設定 -----------------------------------------------------------

# カーネル 11 種類
kernels = [ConstantKernel() * DotProduct() + WhiteKernel(),
           ConstantKernel() * RBF() + WhiteKernel(),
           ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct(),
           ConstantKernel() * RBF(np.ones(x.shape[1])) + WhiteKernel(),
           ConstantKernel() * RBF(np.ones(x.shape[1])) + WhiteKernel() + ConstantKernel() * DotProduct(),
           ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
           ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
           ConstantKernel() * Matern(nu=0.5) + WhiteKernel(),
           ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
           ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
           ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct()]


# 6 モデル構築 ---------------------------------------------------------------

# パラメータ設定
# --- クロスバリデーションのfold数
fold_number = 10

# ハイパーパラメータの設定（線形SVR）
# --- Cの候補
# --- εの候補
linear_svr_cs = 2 ** np.arange(-10, 5, dtype=float)
linear_svr_epsilons = 2 ** np.arange(-10, 0, dtype=float)

# ハイパーパラメータの設定（非線形SVR）
# --- Cの候補
# --- εの候補
# --- γ の候補
nonlinear_svr_cs = 2 ** np.arange(-5, 10, dtype=float)
nonlinear_svr_epsilons = 2 ** np.arange(-10, 0, dtype=float)
nonlinear_svr_gammas = 2 ** np.arange(-20, 10, dtype=float)

# ハイパーパラメータの設定（Oneガウスカーネル）
# --- 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
kernel_number = 2


# モデル構築
if regression_method in ['ols_linear', 'ols_nonlinear']:
    model = LinearRegression()

elif regression_method == 'svr_linear':
    # クロスバリデーションによる C, ε の最適化
    cross_validation = KFold(n_splits=fold_number, random_state=9, shuffle=True)
    gs_cv = GridSearchCV(SVR(kernel='linear'), {'C': linear_svr_cs, 'epsilon': linear_svr_epsilons},
                         cv=cross_validation)  # グリッドサーチの設定
    gs_cv.fit(autoscaled_x, autoscaled_y)  # グリッドサーチ + クロスバリデーション実施
    optimal_linear_svr_c = gs_cv.best_params_['C']  # 最適な C
    optimal_linear_svr_epsilon = gs_cv.best_params_['epsilon']  # 最適な ε
    print('最適化された C : {0} (log(C)={1})'.format(optimal_linear_svr_c, np.log2(optimal_linear_svr_c)))
    print('最適化された ε : {0} (log(ε)={1})'.format(optimal_linear_svr_epsilon, np.log2(optimal_linear_svr_epsilon)))
    model = SVR(kernel='linear', C=optimal_linear_svr_c, epsilon=optimal_linear_svr_epsilon)  # SVRモデルの宣言

elif regression_method == 'svr_gaussian':
    # C, ε, γの最適化
    # 分散最大化によるガウシアンカーネルのγの最適化
    variance_of_gram_matrix = []
    autoscaled_x_array = np.array(autoscaled_x)
    for nonlinear_svr_gamma in nonlinear_svr_gammas:
        gram_matrix = np.exp(
            - nonlinear_svr_gamma * ((autoscaled_x_array[:, np.newaxis] - autoscaled_x_array) ** 2).sum(axis=2))
        variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
    optimal_nonlinear_gamma = nonlinear_svr_gammas[
        np.where(variance_of_gram_matrix == np.max(variance_of_gram_matrix))[0][0]]

    cross_validation = KFold(n_splits=fold_number, random_state=9, shuffle=True)  # クロスバリデーションの分割の設定
    # CV による ε の最適化
    gs_cv = GridSearchCV(SVR(kernel='rbf', C=3, gamma=optimal_nonlinear_gamma),
                         {'epsilon': nonlinear_svr_epsilons},
                         cv=cross_validation)
    gs_cv.fit(autoscaled_x, autoscaled_y)
    optimal_nonlinear_epsilon = gs_cv.best_params_['epsilon']
    # CV による C の最適化
    gs_cv = GridSearchCV(SVR(kernel='rbf', epsilon=optimal_nonlinear_epsilon, gamma=optimal_nonlinear_gamma),
                         {'C': nonlinear_svr_cs},
                         cv=cross_validation)
    gs_cv.fit(autoscaled_x, autoscaled_y)
    optimal_nonlinear_c = gs_cv.best_params_['C']
    # CV による γ の最適化
    gs_cv = GridSearchCV(SVR(kernel='rbf', epsilon=optimal_nonlinear_epsilon, C=optimal_nonlinear_c),
                         {'gamma': nonlinear_svr_gammas},
                         cv=cross_validation)
    gs_cv.fit(autoscaled_x, autoscaled_y)
    optimal_nonlinear_gamma = gs_cv.best_params_['gamma']
    # 結果の確認
    print('最適化された C : {0} (log(C)={1})'.format(optimal_nonlinear_c, np.log2(optimal_nonlinear_c)))
    print('最適化された ε : {0} (log(ε)={1})'.format(optimal_nonlinear_epsilon, np.log2(optimal_nonlinear_epsilon)))
    print('最適化された γ : {0} (log(γ)={1})'.format(optimal_nonlinear_gamma, np.log2(optimal_nonlinear_gamma)))
    # モデル構築
    model = SVR(kernel='rbf', C=optimal_nonlinear_c, epsilon=optimal_nonlinear_epsilon,
                gamma=optimal_nonlinear_gamma)  # SVR モデルの宣言

elif regression_method == 'gpr_one_kernel':
    selected_kernel = kernels[kernel_number]
    model = GaussianProcessRegressor(alpha=0, kernel=selected_kernel)

elif regression_method == 'gpr_kernels':
    # クロスバリデーションによるカーネル関数の最適化
    cross_validation = KFold(n_splits=fold_number, random_state=9, shuffle=True)  # クロスバリデーションの分割の設定
    r2cvs = []  # 空の list。主成分の数ごとに、クロスバリデーション後の r2 を入れていきます
    for index, kernel in enumerate(kernels):
        print(index + 1, '/', len(kernels))
        model = GaussianProcessRegressor(alpha=0, kernel=kernel)
        est_y_in_cv = np.ndarray.flatten(
            cross_val_predict(model, autoscaled_x, autoscaled_y, cv=cross_validation))
        est_y_in_cv = est_y_in_cv * y.std(ddof=1) + y.mean()
        r2cvs.append(r2_score(y, est_y_in_cv))
    optimal_kernel_number = np.where(r2cvs == np.max(r2cvs))[0][0]  # クロスバリデーション後の r2 が最も大きいカーネル関数の番号
    optimal_kernel = kernels[optimal_kernel_number]  # クロスバリデーション後の r2 が最も大きいカーネル関数
    print('クロスバリデーションで選択されたカーネル関数の番号 :', optimal_kernel_number)
    print('クロスバリデーションで選択されたカーネル関数 :', optimal_kernel)

    # モデル構築
    model = GaussianProcessRegressor(alpha=0, kernel=optimal_kernel)  # GPR モデルの宣言


# 7 モデル学習 --------------------------------------------------------

# モデル学習
model.fit(autoscaled_x, autoscaled_y)

# 標準回帰係数
if regression_method in ['ols_linear', 'ols_nonlinear', 'svr_linear']:
    # 回帰係数の取得
    if regression_method == 'svr_linear':
        standard_regression_coefficients = model.coef_.T
    else:
        standard_regression_coefficients = model.coef_

    # データフレーム格納
    standard_regression_coefficients = \
        pd.DataFrame(standard_regression_coefficients,
                     index=x.columns, columns=['standard_regression_coefficients'])


# 8 予測値の取得 --------------------------------------------------------

# 予測値の推定
# --- 推定後に元のスケールに戻す
autoscaled_est_y = model.predict(autoscaled_x)
est_y = autoscaled_est_y * y.std() + y.mean()
est_y = pd.DataFrame(est_y, index=x.index, columns=['est_y'])

# 予測精度の算出
# トレーニングデータのr2, RMSE, MAE
print('r^2 for training data :', r2_score(y, est_y))
print('RMSE for training data :', mean_squared_error(y, est_y, squared=False))
print('MAE for training data :', mean_absolute_error(y, est_y))

# データフレーム格納
# --- 実測値 / 予測値 / 推定誤差
y_for_save = pd.DataFrame(y)
y_for_save.columns = ['actual_y']
y_error_train = y_for_save.iloc[:, 0] - est_y.iloc[:, 0]
y_error_train = pd.DataFrame(y_error_train)
y_error_train.columns = ['error_of_y(actual_y-est_y)']
results_train = pd.concat([y_for_save, est_y, y_error_train], axis=1)


# 9 プロット作成 ------------------------------------------------------------------

# ベースプロット
# --- 実測値 vs. 推定値プロット
plt.rcParams['font.size'] = 12
plt.scatter(y, est_y.iloc[:, 0], c='blue')

# 対角線を作成
y_max = max(y.max(), est_y.iloc[:, 0].max())
y_min = min(y.min(), est_y.iloc[:, 0].min())
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成

# 装飾
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('actual y')
plt.ylabel('est y')
plt.gca().set_aspect('equal', adjustable='box')

# 表示
plt.show()


# 10 クロスバリデーション予測 -------------------------------------------------

# ＜ポイント＞
# - クロスバリデーションで予測値を取得する


# クロスバリデーションによる予測
cross_validation = KFold(n_splits=fold_number, random_state=9, shuffle=True)
autoscaled_est_y_in_cv = cross_val_predict(model, autoscaled_x, autoscaled_y, cv=cross_validation)

# データフレーム格納
# --- 元のスケールに戻す
est_y_in_cv = autoscaled_est_y_in_cv * y.std() + y.mean()
est_y_in_cv = pd.DataFrame(est_y_in_cv, index=x.index, columns=['est_y'])

# 予測精度の確認
# --- r2, RMSE, MAE
print('r^2 in cross-validation :', r2_score(y, est_y_in_cv))
print('RMSE in cross-validation :', mean_squared_error(y, est_y_in_cv, squared=False))
print('MAE in cross-validation :', mean_absolute_error(y, est_y_in_cv))


# プロット作成
plt.rcParams['font.size'] = 12
plt.scatter(y, est_y_in_cv.iloc[:, 0], c='blue')
y_max = max(y.max(), est_y_in_cv.iloc[:, 0].max())
y_min = min(y.min(), est_y_in_cv.iloc[:, 0].min())
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('actual y')
plt.ylabel('est y')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# データフレーム格納
# --- 実測値 / 予測値 / 推定誤差
y_error_in_cv = y_for_save.iloc[:, 0] - est_y_in_cv.iloc[:, 0]
y_error_in_cv = pd.DataFrame(y_error_in_cv)
y_error_in_cv.columns = ['error_of_y(actual_y-est_y)']
results_in_cv = pd.concat([y_for_save, est_y_in_cv, y_error_in_cv], axis=1)  # 結合


# 11 予測データの取得 -------------------------------------------------------

# ＜ポイント＞
# - ADの判定を行うための予測データをデータフレームに準備する


# 予測誤差の取得
# --- 予測誤差 * 予測値の標準偏差
if regression_method in ['gpr_one_kernel', 'gpr_kernels']:
    est_y_pred, est_y_pred_std = model.predict(autoscaled_x_pred, return_std=True)
    est_y_pred_std = est_y_pred_std * y.std()
    est_y_pred_std = pd.DataFrame(est_y_pred_std, x_pred.index, columns=['std_of_est_y'])
else:
    est_y_pred = model.predict(autoscaled_x_pred)

# データフレーム格納
# --- 元のスケールに戻す
est_y_pred = est_y_pred * y.std() + y.mean()
est_y_pred = pd.DataFrame(est_y_pred, x_pred.index, columns=['est_y'])

# 非線形変換を戻す
if regression_method == 'ols_nonlinear':
    x = x_tmp.copy()
    x_pred = x_pred_tmp.copy()
    # 標準偏差が 0 の特徴量の削除
    deleting_variables = x.columns[x.std() == 0]
    x = x.drop(deleting_variables, axis=1)
    x_pred = x_pred.drop(deleting_variables, axis=1)
    # オートスケーリング
    autoscaled_x = (x - x.mean()) / x.std()
    autoscaled_x_pred = (x_pred - x.mean()) / x.std()


# 12 ADによる判定 --------------------------------------------------------------

# ＜ポイント＞
# - 予測値に対してADを適用してAD外のサンプルを把握する

# パラメータ設定
rate_of_training_samples_inside_ad = 0.96

# ハイパーパラメータの設定（ADアルゴリズム）
k_in_knn = 5
ocsvm_nu = 0.04
ocsvm_gamma = 0.1
ocsvm_gammas = 2 ** np.arange(-20, 11, dtype=float)

# AD
if ad_method == 'knn':
    # モデル構築＆学習
    ad_model = NearestNeighbors(n_neighbors=k_in_knn, metric='euclidean')
    ad_model.fit(autoscaled_x)

    # サンプルごとの k 最近傍サンプルとの距離に加えて、k 最近傍サンプルのインデックス番号も一緒に出力されるため、出力用の変数を 2 つに
    # トレーニングデータでは k 最近傍サンプルの中に自分も含まれ、自分との距離の 0 を除いた距離を考える必要があるため、k_in_knn + 1 個と設定
    knn_distance_train, knn_index_train = \
        ad_model.kneighbors(autoscaled_x, n_neighbors=k_in_knn + 1)

    # データフレーム格納
    knn_distance_train = pd.DataFrame(knn_distance_train, index=autoscaled_x.index)

    # 自分以外の k_in_knn 個の距離の平均
    mean_of_knn_distance_train = pd.DataFrame(knn_distance_train.iloc[:, 1:].mean(axis=1),
                                              columns=['mean_of_knn_distance'])

    # トレーニングデータのサンプルの rate_of_training_samples_inside_ad * 100 % が含まれるようにしきい値を設定
    # 距離の平均の小さい順に並び替え
    sorted_mean_of_knn_distance_train = \
        mean_of_knn_distance_train.iloc[:, 0].sort_values(ascending=True)
    ad_threshold = sorted_mean_of_knn_distance_train.iloc[
        round(autoscaled_x.shape[0] * rate_of_training_samples_inside_ad) - 1]

    # トレーニングデータに対して、AD の中か外かを判定
    inside_ad_flag_train = mean_of_knn_distance_train <= ad_threshold

    # 予測用データに対する k-NN 距離の計算
    knn_distance_pred, knn_index_pred = ad_model.kneighbors(autoscaled_x_pred)
    knn_distance_pred = pd.DataFrame(knn_distance_pred, index=x_pred.index)
    ad_index_pred = pd.DataFrame(knn_distance_pred.mean(axis=1), columns=['mean_of_knn_distance'])
    inside_ad_flag_pred = ad_index_pred <= ad_threshold

elif ad_method == 'ocsvm':
    if ad_method == 'ocsvm_gamma_optimization':
        # 分散最大化によるガウシアンカーネルのγの最適化
        variance_of_gram_matrix = []
        autoscaled_x_array = np.array(autoscaled_x)
        for nonlinear_svr_gamma in ocsvm_gammas:
            gram_matrix = np.exp(
                - nonlinear_svr_gamma * ((autoscaled_x_array[:, np.newaxis] - autoscaled_x_array) ** 2).sum(axis=2))
            variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
        optimal_gamma = ocsvm_gammas[np.where(variance_of_gram_matrix == np.max(variance_of_gram_matrix))[0][0]]
        # 最適化された γ
        print('最適化された gamma :', optimal_gamma)
    else:
        optimal_gamma = ocsvm_gamma

    # OCSVM による AD
    ad_model = OneClassSVM(kernel='rbf', gamma=optimal_gamma, nu=ocsvm_nu)  # AD モデルの宣言
    ad_model.fit(autoscaled_x)  # モデル構築

    # トレーニングデータのデータ密度 (f(x) の値)
    data_density_train = ad_model.decision_function(autoscaled_x)
    number_of_support_vectors = len(ad_model.support_)
    number_of_outliers_in_training_data = sum(data_density_train < 0)
    print('\nトレーニングデータにおけるサポートベクター数 :', number_of_support_vectors)
    print('トレーニングデータにおけるサポートベクターの割合 :', number_of_support_vectors / x.shape[0])
    print('\nトレーニングデータにおける外れサンプル数 :', number_of_outliers_in_training_data)
    print('トレーニングデータにおける外れサンプルの割合 :', number_of_outliers_in_training_data / x.shape[0])
    data_density_train = pd.DataFrame(data_density_train, index=x.index, columns=['ocsvm_data_density'])
    data_density_train.to_csv('ocsvm_data_density_train.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意
    # トレーニングデータに対して、AD の中か外かを判定
    inside_ad_flag_train = data_density_train >= 0
    # 予測用データのデータ密度 (f(x) の値)
    ad_index_pred = ad_model.decision_function(autoscaled_x_pred)
    number_of_outliers_in_pred_data = sum(ad_index_pred < 0)
    print('\nテストデータにおける外れサンプル数 :', number_of_outliers_in_pred_data)
    print('テストデータにおける外れサンプルの割合 :', number_of_outliers_in_pred_data / x_pred.shape[0])
    ad_index_pred = pd.DataFrame(ad_index_pred, index=x_pred.index, columns=['ocsvm_data_density'])
    ad_index_pred.to_csv('ocsvm_ad_index_pred.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意
    # 予測用トデータに対して、AD の中か外かを判定
    inside_ad_flag_pred = ad_index_pred >= 0


# 13 次の実験候補の選別 -------------------------------------------------------------

# ＜ポイント＞
# - yの推定値が最大となるサンプルを抽出する


# 予測値の更新
# --- AD外の候補においては負に非常に大きい値を代入（次の候補として選ばれないようにする）
est_y_pred[np.logical_not(inside_ad_flag_pred)] = -10 ** 10

# 次のサンプル
next_sample = x_pred.iloc[est_y_pred.idxmax(), :]
