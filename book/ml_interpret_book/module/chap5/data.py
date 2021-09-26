import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(42)


def generate_simulation_data():

    # パラメータ設定
    # --- インスタンス数
    N = 1000

    # 特徴量の生成
    # --- X0とX1は一様分布から生成
    # --- X2は二項分布から作成
    x0 = np.random.uniform(low=-1, high=1, size=N)
    x1 = np.random.uniform(low=-1, high=1, size=N)
    x2 = np.random.binomial(n=1, p=0.5, size=N)

    # ノイズは正規分布から生成
    epsilon = np.random.normal(loc=0, scale=0.1, size=N)

    # 特徴量をまとめる
    X = np.column_stack([x0, x1, x2])

    # 線形和で目的変数を作成
    y = x0 - 5 * x1 + 10 * x1 * x2 + epsilon

    return train_test_split(X, y, test_size=0.2, random_state=42)

