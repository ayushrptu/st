import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from sklearn.cluster import spectral_clustering


indices = {
    "^GSPC": "S&P 500",
    "^DJI": "Dow Jones Industrial Average",
    "^IXIC": "NASDAQ Composite",
    "^NYA": "NYSE COMPOSITE (DJ)",
    "^XAX": "NYSE AMEX COMPOSITE INDEX",
    "^BUK100P": "Cboe UK 100",
    "^RUT": "Russell 2000",
    "^VIX": "Vix",
    "^FTSE": "FTSE 100",
    "^GDAXI": "DAX PERFORMANCE-INDEX",
    "^FCHI": "CAC 40",
    "^STOXX50E": "ESTX 50 PR.EUR",
    "^N100": "Euronext 100 Index",
    "^BFX": "BEL 20",
    "IMOEX.ME": "MOEX Russia Index",
    "^N225": "Nikkei 225",
    "^HSI": "HANG SENG INDEX",
    "000001.SS": "SSE Composite Index",
    "399001.SZ": "Shenzhen Index",
    "^STI": "STI Index",
    "^AXJO": "S&P/ASX 200",
    "^AORD": "ALL ORDINARIES",
    "^BSESN": "S&P BSE SENSEX",
    "^JKSE": "Jakarta Composite Index",
    "^KLSE": "FTSE Bursa Malaysia KLCI",
    "^NZ50": "S&P/NZX 50 INDEX GROSS",
    "^KS11": "KOSPI Composite Index",
    "^TWII": "TSEC weighted index",
    "^GSPTSE": "S&P/TSX Composite index",
    "^BVSP": "IBOVESPA",
    "^MXX": "IPC MEXICO",
    "^IPSA": "S&P/CLX IPSA",
    "^MERV": "MERVAL",
    "^TA125.TA": "TA-125",
    "^JN0U.JO": "Top 40 USD Net TRI Index"
}


def corr(data_df, sort=None, whitelist=None):
    ps_ = data_df.xs("Close", level=1, axis=1).corr().to_numpy()
    names = np.array(data_df.xs("Close", level=1, axis=1).keys())

    if sort is not None:
        ps_ = ps_[sort, :][:, sort]
        names = names[sort]

    if whitelist is not None:
        wl = [i for i, x in enumerate(names) if x in whitelist]
        ps_ = ps_[wl, :][:, wl]
        names = names[wl]
        ps = np.abs(ps_)

    # k = 7
    # labels = spectral_clustering((1 - ps) ** (1/2), n_clusters=k, eigen_solver="arpack")
    # old_labels = labels.copy()
    # cl_ord = list(dict.fromkeys(labels))
    # for i, j in zip(cl_ord, range(10, 10 * (k + 1), 10)):
    #     labels[labels == i] = j
    # ord = np.argsort(labels)
    # ps = ps[ord, :][:, ord]
    # names = names[ord]
    # labels = labels[ord]
    # anks = [np.sum(labels == i) for i in range(10, 10 * (k + 1), 10)]
    # canks = np.cumsum([0] + anks)

    # from plot import corr
    # corr(names, ps, ps_)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(ps_, vmin=0, vmax=1, cmap="RdBu_r", alpha=0.8)
    ax.set_xticks(np.arange(0, len(names)), names, rotation=45, ha="right")
    ax.set_yticks(np.arange(0, len(names)), names)
    # for c, d in zip(canks, anks):
    #     ax.add_patch(patches.Rectangle((c-0.5, c-0.5), d, d, ec="limegreen", fc="none", linewidth=3))
    fig.colorbar(im, ax=ax)
    fig.suptitle("Pearson Correlation: Stock Indices")
    plt.show()


def volatility(data_df, save=None, do_nan="interp", start=None, save_log_returns=False):
    """
    Compute a local volatility metric using the moving variance of the returns

    Save data with labels: 1 - high volatility, 0 - low volatility

    :param do_nan: "ignore" - nothing
                   "ffill"  - reuse last
                   "interp" - linear interpolation
    """
    if do_nan == "ffill":
        data_df = data_df.ffill()
        if save is not None:
            save = save[:-4] + "_ff" + save[-4:]
    elif do_nan == "interp":
        data_df = data_df.interpolate("linear", limit_area="inside")
        if save is not None:
            save = save[:-4] + "_interp" + save[-4:]

    all_ret = (data_df.xs("Close", level=1, axis=1) / data_df.xs("Close", level=1, axis=1).shift(1)).mean(axis=1)
    all_na = all_ret.isna()
    all_ret = all_ret.loc[~all_na]
    all_vol = all_ret.rolling(200, center=True).var()
    high = all_vol > 5 * 10 ** (-5)
    low = all_vol < 3 * 10 ** (-5)

    for stock in ["^GSPC", "^DJI"]:
        ret = (data_df[stock]["Close"] / data_df[stock]["Close"].shift(1)).loc[~all_na]
        ret_na = ret.isna()
        ret = ret.loc[~ret_na]
        ths_high = high.loc[~ret_na]
        ths_low = low.loc[~ret_na]

        ret_log = ret.apply(np.log)
        vol = ret.rolling(200, center=True).var()
        # high = vol > 1 * 10**(-4)
        # low = vol < 7 * 10**(-5)

        from plot import stock as plt_stock
        plt_stock(data_df[stock]["Close"], ret_log, ths_high, ths_low, all_vol)

        fig, ax = plt.subplots(2, 1, figsize=(18, 10))
        ret_log.mask(~ths_high).plot(ax=ax[0], lw=1, alpha=0.2, label="log returns (high vol)", c="tab:red")
        ret_log.mask(~ths_low).plot(ax=ax[0], lw=1, alpha=0.2, label="log returns (low vol)", c="tab:green")
        ret_log.mask(ths_high | ths_low).plot(ax=ax[0], lw=1, alpha=0.2, label="log returns", c="tab:blue")
        ret_log.rolling(200, center=True).mean().plot(ax=ax[0], lw=1, alpha=1, label="200-day mean", c="tab:orange")
        vol.plot(ax=ax[1], lw=1, alpha=1, logy=True, label="200-day var", c="tab:orange")
        ret.rolling(50, center=True).var().plot(ax=ax[1], lw=1, alpha=0.2, logy=True, label="50-day var", c="tab:red")
        all_vol.plot(ax=ax[1], lw=1, alpha=1, logy=True, label="200-day var (World Indices)", c="tab:blue")
        all_ret.rolling(50, center=True).var().plot(ax=ax[1], lw=1, alpha=0.2, logy=True,
                                                    label="50-day var (World Indices)", c="b")
        ax[0].legend()
        ax[1].legend()
        ax[0].set_title(indices[stock])
        fig.tight_layout()
    plt.show()

    if save is not None:
        if save_log_returns:
            out = (data_df.xs("Close", level=1, axis=1) / data_df.xs("Close", level=1, axis=1).shift(1))
            out = out.loc[~all_na].loc[high | low].apply(np.log)
            out["didx"] = high
        else:
            out = data_df.xs("Close", level=1, axis=1).iloc[:-1, :]
            out["didx"] = (high.astype(int) - low.astype(int))
        out.to_csv(save, sep=";", decimal=",")


def analyze(data_df, **kwargs):
    # Plot
    # data_df.xs("Close", level=1, axis=1).plot(lw=1, title="Close")
    # plt.show()

    # Start date
    # (~data_df.xs("Close", level=1, axis=1).isna()).expanding(1).max().sum(axis=1).plot(title="Number of Indices")
    # plt.show()
    st = (~data_df.xs("Close", level=1, axis=1).isna()).idxmax().to_frame('pos').assign(
        val=lambda d: data_df.xs("Close", level=1, axis=1).lookup(d.pos, d.index))
    print(st.sort_values(by=["pos"]))

    corr(data_df, sort=st.reset_index().sort_values("pos").index)
    # volatility(data_df, **kwargs)


if __name__ == "__main__":
    file = "E:/Daten/Master/data/finance_indices.csv"
    try:
        data_df = pd.read_csv(file, sep=";", decimal=",",
                              header=[0, 1], index_col=0, parse_dates=True, infer_datetime_format=True)
        data_df = data_df.sort_index()
    except FileNotFoundError as e:
        data_df = yf.download(" ".join(indices.keys()), start="1975-01-01", end="2022-08-31", group_by="ticker")
        data_df.to_csv(file, sep=";", decimal=",")

    analyze(data_df, save=None, do_nan="interp", start="1990-01-01")
    # analyze(data_df, save="E:/Daten/Master/data/stock_vol.csv", do_nan="interp", start="1990-01-01")


"""
High vol:
^GSPC    0.000361
^IXIC    0.000383
^NYA     0.000359
^N225    0.000328
^HSI     0.000445

Low vol:
^GSPC    0.000052
^IXIC    0.000076
^NYA     0.000047
^N225    0.000133
^HSI     0.000137
"""
