# ***************************************************************************************
# Book      : Pythonではじめる教師なし学習
# Chapter   : 3.次元削減
# Title     : 3.1 次元削減を行う動機
# Created by: Owner
# Created on: 2020/12/31
# Page      : P67 - P71
# ***************************************************************************************


# ＜次元削減を行う動機＞
# - 機械学習でよく問題となる｢次元の呪い｣に対抗する方策として用いられる
#   --- ｢次元の呪い｣とは特徴量空間が膨大になりすぎて機械学習アルゴリズムが効率的に学習できなくなる現象
#   --- 特徴量空間とは特徴量の数のことを指す
# - 次元削減アルゴリズムは高次元データを低次元空間に射影する
#   --- 冗長な情報を削除しつつ、可能な限り重要な情報を残すようにする
#   --- データを低次元空間に射影すると、ノイズが低減したことを意味する


# ＜目次＞
# 1 MNISTデータのロード
# 2 データセットの確認
# 3 Pandasデータフレームの準備
# 4 データフレームの確認
# 5 画像の表示


# 0 準備 ----------------------------------------------------------------------

# メイン
import gzip
import os
import pickle
import pandas as pd

# 可視化
import matplotlib.pyplot as plt
import seaborn as sns

# カラーパレット
color = sns.color_palette()


# 1 MNISTデータのロード ---------------------------------------------------------

# ＜ポイント＞
# - MNISTデータは以下のデータで構成されている
#   --- 訓練セット(50000)
#   --- 検証セット(10000)
#   --- テストセット(10000)


# パス設定
current_path = os.getcwd()
file = os.path.sep.join(['', 'datasets', 'mnist_data', 'mnist.pkl.gz'])

# データ取得
# --- MNISTデータセット(15790kb)
# --- gzファイルにpickleファイルが保存されている
f = gzip.open(current_path + file, 'rb')
train_set, validation_set, test_set = pickle.load(f, encoding='latin1')
f.close()

# データのセット
# --- 要素[0] : numpy.ndarray(行列)
# --- 要素[1] : numpy.ndarray(ベクトル)
X_train, y_train = train_set[0], train_set[1]
X_validation, y_validation = validation_set[0], validation_set[1]
X_test, y_test = test_set[0], test_set[1]


# 2 データセットの確認 ------------------------------------------------------------

# Train
print("Shape of X_train: ", X_train.shape)
print("Shape of y_train: ", y_train.shape)

# Validation
print("Shape of X_validation: ", X_validation.shape)
print("Shape of y_validation: ", y_validation.shape)

# Test
print("Shape of X_test: ", X_test.shape)
print("Shape of y_test: ", y_test.shape)


# 3 Pandasデータフレームの準備 ------------------------------------------------------

# ＜ポイント＞
# - len()で行列の行数を取得
# - DataFrameとSeriesのインデックスとして使用
# - Trainは0-49999, Validationは50000-59999, Testは60000-69999


# 訓練データ
train_index = range(0, len(X_train))
X_train = pd.DataFrame(data=X_train, index=train_index)
y_train = pd.Series(data=y_train, index=train_index)

# 検証データ
validation_index = range(len(X_train), len(X_train) + len(X_validation))
X_validation = pd.DataFrame(data=X_validation, index=validation_index)
y_validation = pd.Series(data=y_validation, index=validation_index)

# テストデータ
test_index = range(len(X_train) + len(X_validation),
                   len(X_train) + len(X_validation) + len(X_test))
X_test = pd.DataFrame(data=X_test, index=test_index)
y_test = pd.Series(data=y_test, index=test_index)


# 4 データフレームの確認 --------------------------------------------------------------

# データ確認
# --- 訓練データのみ
X_train.head()
y_train.head()

# データ確認
X_train.describe()
y_train.describe()


# 5 画像の表示 -----------------------------------------------------------------------

# 関数定義
# --- MNISTデータのプロット表示
def view_digit(example):
    label = y_train.loc[example]
    image = X_train.loc[example, :].values.reshape([28, 28])
    plt.title('Example: %d  Label: %d' % (example, label))
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.show()


# 画像表示
# --- 1番目
view_digit(0)
