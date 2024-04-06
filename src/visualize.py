import numpy as np
import matplotlib.pyplot as plt
import wandb
from abc import ABC


# plotting and visualizing


class Plotter(ABC):

    def plot_line(self, x, y, label, title, id):
        pass

    def plot_lines(self, x, ys, labels, title, id, silent=False):
        pass


class Plotter_local(Plotter):

    def __init__(self):
        # access matplotlib fig and ax by id
        self.plot_dict = dict()

    def plot_line(self, x, y, label, title, id):
        if id in self.plot_dict.keys():
            fig, ax = self.plot_dict[id]
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            self.plot_dict[id] = (fig, ax)

        ax.plot(x, y, label=label)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        if label != "ground truth":
            plt.show()

    def plot_lines(self, x, ys, labels, title, id, silent=False):
        dt = 0.001 * np.max([np.max(y) - np.min(y) for y in ys])

        if id in self.plot_dict.keys():
            fig, ax = self.plot_dict[id]
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            self.plot_dict[id] = (fig, ax)

        for i in range(len(ys)):
            label, alpha = (labels[i], 1) if len(labels) > i else (None, 0.1)
            ax.plot(x, ys[i] + i * dt, label=label, alpha=alpha)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        if not silent:
            plt.show()

    def combine(self, idxs):
        a = int(np.ceil(np.sqrt(len(idxs))))
        b = int(np.ceil(len(idxs) / a))
        fig, dax = plt.subplots(a, b, figsize=(16, 10))
        for id, cax in zip(idxs, dax.reshape(-1)):
            self.plot_dict[id] = (fig, cax)


class Plotter_wandb(Plotter):

    def plot_line(self, x, y, labels, title, id):
        data = [[x_, y_] for (x_, y_) in zip(x, y)]
        table = wandb.Table(data=data, columns=["x", "y"])
        wandb.log({id: wandb.plot.line(table, "x", "y", title=title)})

    def plot_lines(self, x, ys, labels, title, id, silent=False):
        print(id)
        wandb.log({id: wandb.plot.line_series(x, ys, keys=labels, title=title)})
