
from matplotlib import pyplot as plt


def plot_bar(variables, values, xlabel=None, ylabel=None, title=None):
    fig, ax = plt.subplots()
    ax.barh(variables, values)
    ax.set(xlabel=xlabel, ylabel=ylabel, ylim=(0, None))
    fig.suptitle(title)
    fig.show()
