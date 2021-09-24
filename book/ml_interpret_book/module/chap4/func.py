
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import plot_partial_dependence


def plot_scatter(X, y, xlabel="X", ylabel="Y", title=None):
    fig, ax = plt.subplots()
    sns.scatterplot(X, y, ci=None, alpha=0.3, ax=ax)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    fig.suptitle(title)
    fig.show()


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


def plot_boston_pd(estimator, X, var_name):
    fig, ax = plt.subplots()
    plot_partial_dependence(
        estimator=estimator,
        X=X,
        features=[var_name],
        kind="average",
        ax=ax
    )
    fig.suptitle(f"Partial Dependence Plot ({var_name})")
    fig.show()
