import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


class D:
    def __init__(self, scale):
        self.scale = scale


class Smalldata:
    def __init__(self, data, scale):
        self.data = data
        self.dataset = D(scale)

    def __iter__(self):
        for elem in self.data:
            yield elem

    def __len__(self):
        return len(self.data)


def data_split(self, batch_size=None, valid_perc=None, test_perc=None, use_position=True):
    """
    Create PyTorch dataloaders by batching a time series
    :param batch_size: batch_size, default 1
    :param valid_perc: Percentage of data for the validation set, default 0
    :param test_perc: Percentage of data for the validation set, default 0
    :param use_position: Use self.idxs to inform overlap buffer (else use default seq_len - 1 buffer)
    :return:
    """
    # function setting > class setting > defaults
    batch_size = batch_size if batch_size is not None else self.dataloader_kwargs.get("batch_size", 1)
    valid_perc = valid_perc if valid_perc is not None else self.dataloader_kwargs.get("valid_perc", 0)
    test_perc = test_perc if test_perc is not None else self.dataloader_kwargs.get("valid_perc", 0)

    # Compute length without overlapping sets
    data_len = len(self)
    num_sets = ((valid_perc > 0) + (test_perc > 0))
    new_data_len = data_len - (self.seq_len - 1) * num_sets
    if not hasattr(self, "idxs"):
        use_position = False
    if use_position:
        # on segmented data (e.g. with select_class) buffers are easier
        dt = self.idxs[1:] - self.idxs[:-1]
        loc = np.where(dt > 2)[0]
        gap = dt[loc]
        loc = np.append(np.insert(loc+1, 0, 0), len(self))
        chunk = loc[1:] - loc[:-1]
        assert sum(chunk) == len(self)
        # todo: do something smart here (bin packing etc)
        # estimate gains by half of all
        new_data_len = data_len - ((self.seq_len - 1) * num_sets) / 2
    valid_size = int(new_data_len * valid_perc)
    test_size = int(new_data_len * test_perc)

    def buffer(data_len):
        if use_position:
            reduce = self.seq_len - 1
            # reduce by this but count gaps
            while True:
                ths_loc = np.where((data_len - reduce <= loc) & (loc < data_len))[0]
                if ths_loc.size == 0:
                    data_len = data_len - reduce
                    break
                else:
                    reduce = reduce - gap[ths_loc[-1] - 1]
                    data_len = loc[ths_loc[-1]]
                    if reduce <= 0:
                        break
            return data_len
        else:
            return data_len - (self.seq_len - 1)

    loaders = []
    # No overlaps
    if test_perc > 0:
        idxs = np.arange(data_len-test_size, data_len)
        test = DataLoader(self, sampler=SubsetRandomSampler(idxs), batch_size=batch_size)
        loaders = [test] + loaders
        data_len = data_len - test_size
        data_len = buffer(data_len)
    # else:
    #     loaders = [None] + loaders
    if valid_perc > 0:
        idxs = np.arange(data_len - valid_size, data_len)
        valid = DataLoader(self, sampler=SubsetRandomSampler(idxs), batch_size=batch_size)
        loaders = [valid] + loaders
        data_len = data_len - valid_size
        data_len = buffer(data_len)
    # else:
    #     loaders = [None] + loaders
    idxs = np.arange(0, data_len)
    train = DataLoader(self, sampler=SubsetRandomSampler(idxs), batch_size=batch_size)
    loaders = [train] + loaders

    if num_sets > 0:
        lens = [len(l.sampler.indices) for l in loaders]
        print("data split [%s], %.1f %% discarded as buffer" %
              (", ".join(["%.1f %%" % (100 * l / len(self)) for l in lens]), 100 * (len(self) - sum(lens)) / len(self)))
    return loaders