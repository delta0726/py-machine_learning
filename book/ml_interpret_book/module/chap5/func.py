import matplotlib.pyplot as plt
import seaborn as sns


def plot_scatter(X, y, group, xlabel="X", ylabel="Y", title=None):
    fig, ax = plt.subplots()
    sns.scatterplot(X, y, style=group, ci=None, alpha=0.3, ax=ax)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    fig.suptitle(title)
    fig.show()
