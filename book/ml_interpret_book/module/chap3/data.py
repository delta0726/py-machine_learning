# ******************************************************************************
# Book Name : 機械学習を解釈する技術
# Chapter   : 3 特徴量の重要度を知る
# Module    : simulation_data.py
# Created on: 2021/09/18
# Page      : P57 - P59
# URL       : https://github.com/ghmagazine/ml_interpret_book
# ******************************************************************************

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

from sklearn.model_selection import train_test_split


def generate_simulation_data(N, beta, mu, Sigma):
    X = np.random.multivariate_normal(mu, Sigma, N)
    epsilon = np.random.normal(0, 0.1, N)
    y = X @ beta + epsilon
    return train_test_split(X, y, test_size=0.2, random_state=42)


def plot_scatters(X, y, var_names):

    J = X.shape[1]
    fig, axes = plt.subplots(nrows=1, ncols=J, figsize=(4 * J, 4))

    for d, ax in enumerate(axes):
        sns.scatterplot(X[:, d], y, alpha=0.3, ax=ax)
        ax.set(xlabel=var_names[d],
               ylabel="Y",
               xlim=(X.min() * 1.1, X.max() * 1.1)
               )
        fig.show()
