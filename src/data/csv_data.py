from typing import Dict, Tuple, Union, Optional, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from .data_tools import data_split


class DFData(Dataset):
    """
    Data that is stored as pandas dataframe
    """

    def __init__(self, df, seq_len: int, target_col: List, relevant_cols: List,
                 transform=None, scale=None, **dataloader_kwargs):
        # todo: scale as transform
        self.seq_len = seq_len
        self.transform = transform
        self.dataloader_kwargs = dataloader_kwargs
        self.df = df[relevant_cols].copy()
        self.targ_col = target_col
        self.scale = scale
        if self.scale == "norm-all":
            # self.std_all = self.df.std(axis=0).to_numpy().astype(np.float32)  # overall std is inaccurate due to trend
            self.std_all = self.df.rolling(self.seq_len).std().mean(axis=0).to_numpy().astype(np.float32)
        elif self.scale == "norm-all-detrend":
            stds = []
            for idx in range(len(self)):
                x = self.df.iloc[idx: self.seq_len + idx].to_numpy()
                x_ = np.arange(x.shape[0])
                beta = np.array([np.corrcoef(x_, x[:, i])[0, 1] for i in range(x.shape[1])]) * x.std(axis=0) / x_.std(
                    axis=0)
                alpha = x.mean(axis=0) - beta * x_.mean(axis=0)
                stds += [(x - alpha - beta * x_[:, None]).std(axis=0)]
            self.std_all = np.mean(stds, axis=0).astype(np.float32)

    def __getitem__(self, idx: int):
        x = self.df.iloc[idx: self.seq_len + idx].to_numpy().astype(np.float32)

        if self.scale is not None:
            if type(self.scale) == str:
                if self.scale in ["norm-detrend", "norm-all-detrend"]:
                    # detrend
                    x_ = np.arange(x.shape[0]).astype(np.float32)
                    beta = np.array([np.corrcoef(x_, x[:, i])[0, 1].astype(np.float32) for i in range(x.shape[1])]) * x.std(axis=0) / x_.std(axis=0)
                    alpha = x.mean(axis=0) - beta * x_.mean(axis=0)
                    x = x - alpha - beta * x_[:, None]
                if self.scale in ["norm", "norm-detrend"]:
                    x = x / x.std(axis=0)
                    mn = x.mean(axis=0)
                    x = x - mn
                elif self.scale in ["norm-all", "norm-all-detrend"]:
                    x = x / self.std_all
                    mn = x.mean(axis=0)
                    x = x - mn
                elif self.scale in ["scale", "scale-detrend"]:
                    x = x / x.std(axis=0)
                    mn = x.mean(axis=0)
                else:
                    raise ValueError("unsupported scaling")
            else:
                x = self.scale * x
        else:
            mn = None
        if self.transform is not None:
            return self.transform(x)
        else:
            if self.scale is not None:
                return {"x": x, "y": x.copy(), "mean": mn[None, :]}
            else:
                return {"x": x, "y": x.copy()}

    def __len__(self) -> int:
        return len(self.df.index) - (self.seq_len - 1)

    def dataloader(self, **kwargs):
        return data_split(self, **kwargs)


class CSVData(DFData):
    def __init__(self, file_path, seq_len: int, target_col: List, relevant_cols: List,
                 # scaling=None, scaled_cols=None,
                 transform=None, **dataloader_kwargs):
        """
        A dataset that takes a CSV file and properly batches for use in training/eval a PyTorch model
        :param file_path: The path (oir list of paths) to the CSV file you wish to use.
        :param seq_len: This is the length of the historical time series data you wish to utilize, e.g.  for forecasting
        :param relevant_cols: Supply column names you wish to predict in the forecast (others will not be used)
        :param target_col: The target column or columns you to predict. If you only have one still use a list ['cfs']
        :param csv_weights: list of ints - if not None, train on more copies of datasets with high weight.
        :param scaling: (highly reccomended) If provided should be a subclass of sklearn.base.BaseEstimator
        and sklearn.base.TransformerMixin) i.e StandardScaler,  MaxAbsScaler, MinMaxScaler, etc) Note without
        a scaler the loss is likely to explode and cause infinite loss which will corrupt weights
        :param dataloader_kwargs, parameters for dataloader, e.g. test_perc, val_perc, batch_size
        """
        print("--------------------Reading error -------------------")
        df = pd.read_csv(file_path, sep=";", decimal=",")
        print("--------------------Reading error -------------------")

        super().__init__(df, seq_len, target_col, relevant_cols, transform, **dataloader_kwargs)

        # if scaled_cols is None:
        #     scaled_cols = relevant_cols
        #
        # if scaling is not None:
        #     print("scaling now")
        #     self.scale = scaling.fit(self.df[scaled_cols])
        #     temp_df = self.scale.transform(self.df[scaled_cols])
        #
        #     # We define a second scaler to scale the end output
        #     # back to normal as models might not necessarily predict
        #     # other present time series values.
        #     targ_scale_class = self.scale.__class__
        #     self.targ_scaler = targ_scale_class()
        #     self.df[target_col] = self.targ_scaler.fit_transform(self.df[target_col])
        #
        #     self.df[scaled_cols] = temp_df

        if (len(self.df) - self.df.count()).max() != 0:
            print("Error nan values detected in data. Please run interpolate ffill or bfill on data")

    # def inverse_scale(
    #     self, result_data: Union[torch.Tensor, pd.Series, np.ndarray]
    # ) -> torch.Tensor:
    #     """Un-does the scaling of the data
    #
    #     :param result_data: The data you want to unscale can handle multiple data types.
    #     :type result_data: Union[torch.Tensor, pd.Series, np.ndarray]
    #     :return: Returns the unscaled data as PyTorch tensor.
    #     :rtype: torch.Tensor
    #     """
    #     if not hasattr(self, "targ_sacler"):
    #         return result_data
    #
    #     if isinstance(result_data, pd.Series) or isinstance(
    #         result_data, pd.DataFrame
    #     ):
    #         result_data_np = result_data.values
    #     elif isinstance(result_data, torch.Tensor):
    #         if len(result_data.shape) > 2:
    #             result_data = result_data.permute(2, 0, 1).reshape(result_data.shape[2], -1)
    #             result_data = result_data.permute(1, 0)
    #         result_data_np = result_data.numpy()
    #     elif isinstance(result_data, np.ndarray):
    #         result_data_np = result_data
    #     else:
    #         print(type(result_data))
    #         raise ValueError()
    #     return torch.from_numpy(self.targ_scaler.inverse_transform(result_data_np))


class LabeledCSVData(DFData):
    def __init__(self, file_path, seq_len: int, target_col: List, relevant_cols: List, class_col="didx",
                 transform=None, scale=None, select_class=None, class_weights=None, force_same_class=True, **dataloader_kwargs):
        """
        A dataset that takes a CSV file and properly batches for use in training/eval a PyTorch model
        contains class labels for each time step to allow for different classes.
        Think for example high/low volatility in log returns
        """
        df = pd.read_csv(file_path, sep=";", decimal=",", index_col=0, parse_dates=True, infer_datetime_format=True)

        start = df[relevant_cols].isna().any(axis=1).idxmin()
        if start > df.index[0]:
            print("Discarding %d %% of data due to different start of time series" % (100 * sum(df.index < start) / len(df)))
            df = df[df.index > start]

        if force_same_class:
            self.didx = df[class_col]
            # does didx change here
            const = (self.didx.rolling(seq_len).max().shift(-seq_len+1) ==
                     self.didx.rolling(seq_len).min().shift(-seq_len+1))
        else:
            # > 50 % ?
            self.didx = df[class_col].rolling(seq_len).median().iloc[(seq_len - 1):]
            self.didx[(self.didx == 0.5) | (self.didx == -0.5)] = 0
            const = ~self.didx.isna()
            const.iloc[-(seq_len-1):] = False

        # Filter class label
        if select_class:
            self.idxs = np.where(const & (self.didx == select_class))[0]
        elif class_weights is not None:
            # more classes
            classes = self.didx.unique()
            if len(classes) != len(class_weights):
                raise ValueError("Not all classes have class weights")
            idxs_cls = [np.where(const & (self.didx == cls))[0] for cls in classes]
            self.idxs = np.array([x for i, w in enumerate(class_weights) for x in w * list(idxs_cls[i])])
        else:
            self.idxs = np.where(const)[0]
        self.len = len(self.idxs)

        c_pre = self.didx.iloc[:-(seq_len - 1)].value_counts()
        if class_weights is not None:
            c_pre * np.array(class_weights)
        c_post = self.didx[self.idxs].value_counts()

        print("Discarding %d %% of labeled data due to changing class label [%s]" %
              (100 * (1 - c_post.sum() / c_pre.sum()),
               ", ".join(["%d: %.1f %%" % (k, 100 * v) for k, v in dict(1 - c_post / c_pre).items()])))
        print("Class distribution [%s]" %
              (", ".join(["%d: %d (%.1f %%)" % (k, self.len * v, 100 * v) for k, v in dict(c_post / c_post.sum()).items()])))
        len(self.idxs)
        super().__init__(df, seq_len, target_col, relevant_cols, transform, scale, **dataloader_kwargs)

        if (len(self.df) - self.df.count()).max() != 0:
            print("Error nan values detected in data. Please run interpolate ffill or bfill on data")

    def __getitem__(self, idx: int):
        nidx = self.idxs[idx]
        out = super().__getitem__(nidx)
        out["didx"] = self.didx[nidx]
        return out

    def __len__(self) -> int:
        return self.len


class CSVsData(CSVData):
    def __init__(self, file_paths: List, csv_weights=None, **kwargs):
        """
        A dataset that takes multiple CSV files and properly batches for use in training/eval a PyTorch model
        :param file_paths: list of str - paths to the CSV files you wish to use.
        :param csv_weights: list of ints - if not None, train on more copies of datasets with high weight.

        :param **kwargs: dataset / dataloader params
        """
        kw = ["seq_len", "target_col", "relevant_cols", "scaling", "scaled_cols", "transform"]
        data_kwargs = {k: w for k, w in kwargs.items() if k in kw}
        dataloader_kwargs = {k: w for k, w in kwargs.items() if k not in kw}

        self.data = [CSVData(fp, **data_kwargs) for fp in file_paths]
        if csv_weights is not None:
            self.data_ankers = np.cumsum([0] + [reps * len(d) for d, reps in zip(self.data, csv_weights)])
        else:
            self.data_ankers = np.cumsum([0] + [len(d) for d in self.data])
        self.dataloader_kwargs = dataloader_kwargs

    def __getitem__(self, idx: int):
        didx = np.argmax(self.data_ankers > idx) - 1
        dataset = self.data[didx]
        out = dataset[(idx - self.data_ankers[didx]) % len(dataset)]
        out["didx"] = didx
        return out

    def __len__(self) -> int:
        return self.data_ankers[-1]

    def data_splits(self, batch_size=None, valid_perc=None, test_perc=None):
        """
        Create PyTorch dataloaders by batching a time series
        :param batch_size: batch_size, default 1
        :param valid_perc: Percentage of data for the validation set, default 0
        :param test_perc: Percentage of data for the validation set, default 0
        :return:
        """
        # function setting > class setting > defaults
        batch_size = batch_size if batch_size is not None else self.dataloader_kwargs.get("batch_size", 1)
        valid_perc = valid_perc if valid_perc is not None else self.dataloader_kwargs.get("valid_perc", 0)
        test_perc = test_perc if test_perc is not None else self.dataloader_kwargs.get("valid_perc", 0)

        loaders_idxs = [[], [], []]
        num_sets = ((valid_perc > 0) + (test_perc > 0))
        # Compute length without overlapping sets
        for i, d in enumerate(self.data):
            sub_ankers = np.arange(self.data_ankers[i], self.data_ankers[i+1], len(d))
            for start in sub_ankers:
                data_len = len(d)

                new_data_len = data_len - (d.seq_len - 1) * num_sets
                valid_size = int(new_data_len * valid_perc)
                test_size = int(new_data_len * test_perc)

                # No overlaps
                if test_perc > 0:
                    loaders_idxs[2] += list(start + np.arange(data_len - test_size, data_len))
                    data_len = data_len - test_size - (d.seq_len - 1)
                if valid_perc > 0:
                    loaders_idxs[1] += list(start + np.arange(data_len - valid_size, data_len))
                    data_len = data_len - valid_size - (d.seq_len - 1)
                loaders_idxs[0] += list(start + np.arange(0, data_len))

        loaders = []
        if test_perc > 0:
            test = DataLoader(self, sampler=SubsetRandomSampler(loaders_idxs[2]), batch_size=batch_size)
            loaders = [test] + loaders
        if valid_perc > 0:
            valid = DataLoader(self, sampler=SubsetRandomSampler(loaders_idxs[1]), batch_size=batch_size)
            loaders = [valid] + loaders
        train = DataLoader(self, sampler=SubsetRandomSampler(loaders_idxs[0]), batch_size=batch_size)
        loaders = [train] + loaders

        return loaders

    def dataloader(self, **kwargs):
        return self.data_splits(**kwargs)
