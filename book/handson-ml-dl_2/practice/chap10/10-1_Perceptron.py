# ******************************************************************************
# Title     : Sklearn, Keras, TensorFlowによる実践機械学習
# Chapter   : 10 人工ニューラルネットワークとKerasの初歩
# Theme     : 1 パーセプトロン
# URL       : https://github.com/ageron/handson-ml2
# Date      : 2021/10/17
# Page      : P285 - P288
# ******************************************************************************


# 0 準備
# 1 パーセプトロン学習
# 2 分類結果の可視化


# 0 準備 -------------------------------------------------------------

# ライブラリ
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

# データロード
iris = load_iris()
X = iris.data[:, (2, 3)]
y = (iris.target == 0).astype(np.int)

# データサイズ
X.shape
y.shape


# 1 パーセプトロン学習 --------------------------------------------------

# ＜ポイント＞
# - パーセプトロン学習はANNアーキテクチャのの最も単純なものの1つ
# - Xとyはともに数値で個々の接続部分に重みが与えられる


# インスタンス生成
per_clf = Perceptron(max_iter=1000, tol=1e-3, random_state=42)

# 学習
per_clf.fit(X, y)

# 確認
print(per_clf)
vars(per_clf)

# 予測（インサンプル）
y_pred_in = per_clf.predict(X)
y_pred_in

# 予測（アウトサンプル）
y_pred_out = per_clf.predict([[2, 0.5]])
y_pred_out


# 2 分類結果の可視化 ------------------------------------------------------

# ＜ポイント＞
# - パーセプトロン学習による分類イメージを可視化
#   --- 書籍には記述なし


a = -per_clf.coef_[0][0] / per_clf.coef_[0][1]
b = -per_clf.intercept_ / per_clf.coef_[0][1]

axes = [0, 5, 0, 2]

x0, x1 = np.meshgrid(
        np.linspace(axes[0], axes[1], 500).reshape(-1, 1),
        np.linspace(axes[2], axes[3], 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]
y_predict = per_clf.predict(X_new)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[y==0, 0], X[y==0, 1], "bs", label="Not Iris-Setosa")
plt.plot(X[y==1, 0], X[y==1, 1], "yo", label="Iris-Setosa")

plt.plot([axes[0], axes[1]], [a * axes[0] + b, a * axes[1] + b], "k-", linewidth=3)
custom_cmap = ListedColormap(['#9898ff', '#fafab0'])

plt.contourf(x0, x1, zz, cmap=custom_cmap)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="lower right", fontsize=14)
plt.axis(axes)
plt.show()


# 3 活性化関数の定義 ----------------------------------------------------

# シグモイド関数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ReLU関数
def relu(z):
    return np.maximum(0, z)

# 導関数
def derivative(f, z, eps=0.000001):
    return (f(z + eps) - f(z - eps))/(2 * eps)


# 4 活性化関数の可視化 ---------------------------------------------------

# 出力範囲の定義
z = np.linspace(-5, 5, 200)

# プロットサイズの指定
plt.figure(figsize=(11,4))

# プロット1
# --- 活性化関数
plt.subplot(121)
plt.plot(z, np.sign(z), "r-", linewidth=1, label="Step")
plt.plot(z, sigmoid(z), "g--", linewidth=2, label="Sigmoid")
plt.plot(z, np.tanh(z), "b-", linewidth=2, label="Tanh")
plt.plot(z, relu(z), "m-.", linewidth=2, label="ReLU")
plt.grid(True)
plt.legend(loc="center right", fontsize=14)
plt.title("Activation functions", fontsize=14)
plt.axis([-5, 5, -1.2, 1.2])

# プロット2
# ---
plt.subplot(122)
plt.plot(z, derivative(np.sign, z), "r-", linewidth=1, label="Step")
plt.plot(0, 0, "ro", markersize=5)
plt.plot(0, 0, "rx", markersize=10)
plt.plot(z, derivative(sigmoid, z), "g--", linewidth=2, label="Sigmoid")
plt.plot(z, derivative(np.tanh, z), "b-", linewidth=2, label="Tanh")
plt.plot(z, derivative(relu, z), "m-.", linewidth=2, label="ReLU")
plt.grid(True)
#plt.legend(loc="center right", fontsize=14)

# タイトル
plt.title("Derivatives", fontsize=14)
plt.axis([-5, 5, -0.2, 1.2])

# プロット表示
plt.show()
