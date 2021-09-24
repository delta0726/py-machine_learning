import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(42)


def generate_simulation_data2():

    # パラメータ設定
    # --- インスタンス数
    N = 1000

    # 一様分布から特徴量を生成
    X = np.random.uniform(low=-np.pi * 2, high=np.pi * 2, size=[N, 2])
    epsilon = np.random.normal(loc=0, scale=0.1, size=N)

    # yはsin関数で変換する
    y = 10 * np.sin(X[:, 0]) + X[:, 1] + epsilon

    return train_test_split(X, y, test_size=0.2, random_state=42)
