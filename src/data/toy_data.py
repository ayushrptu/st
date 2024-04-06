import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from .data_tools import data_split
from config import DATA_FOLDER


def gp_likelihood(y, kernel, x_len=None):
    """
    Compute log likelihood of a sample wrt the GP, assuming no further noise
    ! do not use !

    :param y: sampled GP of shape len x num channels
    :param kernel: sklearn kernel
    :param x_len: length of equidistant sampling, if sampled from this class: 0.05 * seq_len
    """
    c = 0
    for ch in range(y.shape[1]):
        y_ths = y[:, ch]
        n = y_ths.size
        if x_len is None:
            x_len = 0.05 * n
        X = np.linspace(0, x_len, n).reshape(-1, 1)
        # y_ths_mean = np.zeros(y_ths.shape)
        R = kernel(X)
        R_sign, R_logdet = np.linalg.slogdet(R)
        if R_sign <= 0:
            R_sign, R_logdet = torch.slogdet(torch.from_numpy(R))
            assert R_sign > 0
            R_logdet = R_logdet.numpy()
        c += (- 0.5 * y_ths.T @ R @ y_ths - 0.5 * R_logdet - n / 2 * np.log(2 * np.pi)).item()
    return c / y.shape[1]


class ToyData(Dataset):
    def __init__(self, n=1000, seq_len=64, channels=1, noise=0, transform=None, max_chunk=5000, **dataloader_kwargs):
        """
        A synthetic dataset with samples from a gaussian process with rbf kernel
        batched the same as real data, with overlapping of consecutive samples
        :param seq_len: length of the time series data
        :param n: number of data points

        :param dataloader_kwargs, parameters for dataloader, e.g. test_perc, val_perc, batch_size
        """
        self.n = n
        self.channels = channels
        self.seq_len = seq_len
        # mean = 0, std = 1
        self.gp = GaussianProcessRegressor(RBF() + WhiteKernel(noise))
        self.ns = (n // max_chunk) * [max_chunk]
        if n % max_chunk != 0:
            self.ns += [n % max_chunk]
        self.X = [np.linspace(0, 0.05 * n, n).reshape(-1, 1) for n in self.ns]
        self.samples = [self.gp.sample_y(X, self.channels, random_state=i).astype(np.float32) for i, X in enumerate(self.X)]
        self.samples = np.concatenate(self.samples, axis=0)

        # corr: AR-1
        for c in range(1, self.channels):
            self.samples[:, c] = 0.8 * self.samples[:, c-1] + 0.2 * self.samples[:, c]
        self.transform = transform
        self.dataloader_kwargs = dataloader_kwargs

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        y_mean, y_std = self.gp.predict(self.X[(0, -1), :], return_std=True)
        for idx, single_prior in enumerate(self.samples.T):
            ax.plot(
                single_prior,
                linestyle="--",
                alpha=0.7,
                label=f"Sampled function #{idx + 1}",
            )
        ax.plot(ax.get_xlim(), y_mean, color="black", label="Mean")
        ax.fill_between(
            ax.get_xlim(),
            y_mean - y_std,
            y_mean + y_std,
            alpha=0.1,
            color="black",
            label=r"$\pm$ 1 std. dev.",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_ylim([-3, 3])

    def __getitem__(self, idx: int):
        x = self.samples[idx: self.seq_len + idx, :]
        if self.transform is not None:
            return self.transform(x)
        else:
            return {"x": x, "y": x.copy()}

    def __len__(self):
        return self.n - (self.seq_len - 1)

    def likelihood(self, y, x_len=None):
        return gp_likelihood(y, self.gp.kernel, x_len)

    def dataloader(self, **kwargs):
        return data_split(self, **kwargs)


class FinToyData(Dataset):
    def __init__(self, seq_len=256, scale=None, transform=None, n=320, **kwargs):
        """
        Stock crash with HMSM
        """
        self.n = n
        self.seq_len = seq_len
        self.transform = transform
        self.scale = scale
        self.dkwargs = kwargs

        rho = np.array([[1, 0.98, 0.97, 0.42, 0.81],
                        [0.98, 1, 0.9, 0.43, 0.73],
                        [0.97, 0.9, 1, 0.36, 0.92],
                        [0.42, 0.43, 0.36, 1, 0],
                        [0.81, 0.73, 0.92, 0, 1]])
        rho -= 0.01 * (1 - np.eye(5))
        sig1 = 0.0083
        sig2 = 0.01
        pc = 0.15
        r1 = np.random.multivariate_normal(0.0001 * np.ones(5), sig1**2 * rho, size=(self.n, int(0.75 * self.seq_len)))
        rc = np.log(1 - pc * np.ones((self.n, 1, 5)))
        r2 = np.random.multivariate_normal(np.zeros(5), sig2**2 * rho, size=(self.n, int(0.75 * self.seq_len)))
        self.data = np.cumprod(np.exp(np.concatenate([r1, rc, r2], axis=1)), axis=1).astype(np.float32)
        self.s = np.random.choice(int(0.25 * self.seq_len), replace=True, size=self.n)

    def __getitem__(self, idx: int):
        x = self.data[idx, self.s[idx]: self.s[idx] + self.seq_len]

        if self.scale is not None:
            if self.scale in ["norm-detrend"]:
                # detrend
                x_ = np.arange(x.shape[0]).astype(np.float32)
                beta = np.array([np.corrcoef(x_, x[:, i])[0, 1].astype(np.float32) for i in range(x.shape[1])]) * x.std(axis=0) / x_.std(axis=0)
                alpha = x.mean(axis=0) - beta * x_.mean(axis=0)
                x = x - alpha - beta * x_[:, None]
            if self.scale in ["norm", "norm-detrend"]:
                x = x / x.std(axis=0)
                mn = x.mean(axis=0)
                x = x - mn
            elif self.scale in ["scale", "scale-detrend"]:
                x = x / x.std(axis=0)
                mn = x.mean(axis=0)
            else:
                raise ValueError("unsupported scaling")
        else:
            mn = None

        if self.transform is not None:
            return self.transform(x)
        else:
            if self.scale is not None:
                return {"x": x, "y": x.copy(), "mean": mn[None, :]}
            else:
                return {"x": x, "y": x.copy()}

    def __len__(self):
        return self.n

    def dataloader(self, **kwargs):
        self.dkwargs.pop("valid_perc")
        self.dkwargs.pop("test_perc")
        return None, None, DataLoader(self, **kwargs, **self.dkwargs)


def create_toydata(n=50000, channels=1):
    # Create and save toy data
    hd = ["x"] if channels == 1 else ["x_%d" % i for i in range(channels)]
    data = ToyData(n=n, channels=channels)
    pd.DataFrame(data.samples).to_csv(DATA_FOLDER + "/toy_data_big_c%d.csv" % channels,
                                      index=None, header=hd, sep=";", decimal=",")
    data = ToyData(n=n, noise=0.5, channels=channels)
    pd.DataFrame(data.samples).to_csv(DATA_FOLDER + "/toy_data_noise_big_c%d.csv" % channels,
                                      index=None, header=hd, sep=";", decimal=",")


def test_toydata(n=1000):
    import matplotlib.pyplot as plt

    # Create and compute statistics for toy data
    data = ToyData(n, seq_len=128)
    lks = []
    for i in range(0, n, 100):
        sample = data[i]["x"]

        fig, ax = plt.subplots()
        ax.set_axis_off()
        ax.plot(sample)
        plt.show()

    # mean -350, in ~ [-6000, 2000], exponential increase
    print("mean likelihood: %.2f" % (sum(lks) / len(lks)))
    plt.hist(lks)
    plt.show()
