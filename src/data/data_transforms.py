import numpy as np


class Mask_TS:
    """ Pre-processing of a dataset by masking a time series for MLM """
    def __init__(self, cloze_len=1, cloze_perc=0.15, mask_rand_none_split=(0.8, 0.1, 0.1),
                 mask_val=0, rand_mean=0, rand_std=1):
        """
        :param cloze_len, size of non-overlapping chunks that may be masked
        :param cloze_perc, percentage of chuncks to be used for training / masking
        :param mask_rand_none_split: = (mask_perc, rand_perc, none_perc)
               percentage of clozed chunks to mask, randomize or ignore

        :param mask_val: value at masked positions
        :param rand_mean, rand_std: parameters of iid normal values at random positions
        """
        assert sum(mask_rand_none_split) == 1, "mask split has to sum up to 1"

        self.cloze_len = cloze_len
        self.cloze_perc = cloze_perc
        self.mask_perc, self.rand_perc, self.none_perc = mask_rand_none_split
        self.weights = [1 - self.cloze_perc] + [self.cloze_perc * p for p in mask_rand_none_split]

        self.mask_val = mask_val
        self.rand_mean = rand_mean
        self.rand_std = rand_std

    def __call__(self, ts):
        """
        :param ts: time series array of shape LxC - where L=length, N=number of channels
        """
        # 0: no cloze, 1: mask, 2: rand, 3: none
        clozed_chunks = np.random.choice(4, size=int(np.ceil(ts.shape[0] / self.cloze_len)), p=self.weights)
        mask = np.repeat(clozed_chunks, self.cloze_len, axis=0)[:ts.shape[0]]
        x = ts.copy()
        x[mask == 1] = self.mask_val
        x[mask == 2] = np.random.normal(self.rand_mean, self.rand_std, size=(np.sum(mask == 2), ts.shape[1]))
        return {"x": x, "y": ts, "mask": mask}


class Gaussian_Noise:
    """
    Pre-processing of a dataset by adding iid gaussian noise to a time series
    If no std is given it will be adaptively set half of the sample std.
    """
    def __init__(self, mean=0, std=None):
        """
        :param mean: int / array-like: mean of noise
        :param std: int / array-like / None: std of noise, default half of sample std
        """
        self.mean = mean
        self.std = std

    def __call__(self, ts):
        """
        :param ts: time series array of shape LxC - where L=length, N=number of channels
        """
        seq_len, num_channels = ts.shape
        if self.std is None:
            stds = np.std(ts, axis=0) / 2
        else:
            stds = num_channels * [self.std] if not hasattr(self.mean, "__getitem__") else self.std
        means = num_channels * [self.mean] if not hasattr(self.mean, "__getitem__") else self.mean
        noise = np.stack([np.random.normal(mean, std, size=seq_len) for mean, std in zip(means, stds)], axis=1).astype(ts.dtype)
        return {"x": ts + noise, "y": ts, "mask": np.zeros(ts.shape, dtype=ts.dtype)}

