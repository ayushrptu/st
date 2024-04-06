import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as plticker


def corr(data):
    cor = np.corrcoef(data.to_numpy().T)
    names = data.columns
    title = "Correlation of features"
    locs = [-0.5, 7.5, 8.5, 11.5]

    fig, ax = plt.subplots(figsize=(10, 8))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(cor, vmin=-1, vmax=1)

    if locs is not None:
        for i in range(len(locs) - 1):
            ax.plot([locs[i], locs[i], locs[i+1], locs[i+1], locs[i]],
                    [locs[i+1], locs[i], locs[i], locs[i+1], locs[i+1]], c="red")

    loc = plticker.FixedLocator(range(cor.shape[0]))  # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    ax.set_xticklabels(names, rotation=-30)
    ax.set_yticklabels(names)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title(title)

    for i, j in itertools.product(range(cor.shape[0]), range(cor.shape[1])):
        ax.text(j, i, "%.2f" % cor[i, j], ha="center", va="center")

    plt.show()


def plot_ts(data):
    fig, ax = plt.subplots(3, 1, figsize=(10, 9))
    data.loc[:, "Masse Destillat": "Masse Sumpf"].plot(ax=ax[0], sharex=True)
    ax[0].plot(data.loc[:, "Masse Destillat": "Masse Sumpf"].sum(axis=1), label="Masse", alpha=0.2)
    ax[0].legend()
    data.loc[:, "T101": "T114"].plot(ax=ax[1], sharex=True)
    data[["PIC101"]].plot(ax=ax[2], sharex=True)
    ax[1].legend(loc="upper right")
    ax[2].set_xticklabels(["%d:00" % x for x in range(11, 16)], rotation=0, ha="center")
    plt.tight_layout()
    plt.show()


def info(data):
    pd.set_option("display.max.columns", None)
    pd.set_option("display.precision", 2)
    data.head()
    data.info()
    data.describe()


if __name__ == "__main__":
    data = pd.read_csv("../data/data.csv", sep=";", decimal=",")
    data["Time"] = pd.to_datetime(data["Time"])
    data.set_index("Time", inplace=True)
    info(data)
    plot_ts(data)
    # corr(data)
