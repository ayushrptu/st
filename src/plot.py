import itertools
import random

import librosa
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import norm
from sphfile import SPHFile

from config import model_config, DATA_FOLDER
from train_eval import load_net_and_data, load_data
from style_transfer import evaluate_style_transfer
from data import Smalldata
from data.speech import plot_sp

matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['svg.fonttype'] = 'none'
cm = 1/2.54


# TIM spectro
def spectro():
    f = DATA_FOLDER + "TIMIT/TEST/DR1/FAKS0/SA2.WAV"
    f = DATA_FOLDER + "TIMIT/TEST/DR1/MSTK0/SA2.WAV"
    sph = SPHFile(f)
    samples = sph.content
    spectrogram = np.abs(librosa.stft(samples.astype(float), n_fft=512))
    print(spectrogram.shape)

    fig, ax = plt.subplots(figsize=(6 * cm, 6 * cm))
    data_log = np.log(spectrogram)
    ax.imshow(data_log, cmap="Greys", vmin=np.percentile(data_log, 0), vmax=np.percentile(data_log, 99),
              origin="lower")
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Seconds')
    ax.set_xticks([0, 103, 206, 309])
    ax.set_yticks([])
    ax.set_xticklabels(["0", "1", "2", "3"])
    fig.savefig("exspect1.svg")
    plt.show()


# FIN corr
def corr(name, ps, ps_):
    names = [r"\ii{%s}" % n[1:] for n in name]
    fig, ax = plt.subplots(figsize=(5 * cm, 4 * cm))
    im = ax.imshow(ps, vmin=0, vmax=1, cmap="RdBu_r", alpha=0.8)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(0, len(names)), names, ha="left", rotation=90)
    ax.set_yticks(np.arange(0, len(names)), names)
    for (j, i), label in np.ndenumerate(ps):
        tx = ax.text(i, j, "\\fn %.2f" % label, ha='left', va='center', fontsize=5)
        tx.set_position((tx.get_position()[0] - 0.4, tx.get_position()[1]))
    cax = fig.colorbar(im, ax=ax)
    cax.set_label(r"\$\abs{\rho}\$", rotation=0, labelpad=10)
    cax.set_ticks([0, 0.5, 1])
    fig.savefig("corr5.svg")
    plt.show()


# FIN vol
def stock(price, ret_log, ths_high, ths_low, all_vol):
    fig, ax = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [2.5, 1, 1]}, figsize=(15 * cm, 10 * cm))
    price[:-50].plot(ax=ax[0], lw=1, c="C0")
    ret_log.mask(~ths_high)[:-50].plot(ax=ax[1], lw=1, alpha=0.5, label="log returns (high vol)", c="tab:red")
    ret_log.mask(~ths_low)[:-50].plot(ax=ax[1], lw=1, alpha=0.5, label="log returns (low vol)", c="tab:green")
    ret_log.mask(ths_high | ths_low)[:-50].plot(ax=ax[1], lw=1, alpha=0.5, label="log returns", c="tab:blue")
    all_vol[:-150].plot(ax=ax[2], lw=1, logy=True, label="50-day var (World Indices)", c="C0", rot=0)
    ax[0].set_ylabel("Price")
    ax[1].set_ylabel("Log-Returns")
    ax[2].set_ylabel("Volatility")
    ax[0].set_xticks([], minor=True)
    ax[1].set_yticks([-0.2, 0, 0.2])
    ax[0].spines.right.set_visible(False)
    ax[0].spines.top.set_visible(False)
    ax[1].spines.right.set_visible(False)
    ax[1].spines.top.set_visible(False)
    ax[2].spines.right.set_visible(False)
    ax[2].spines.top.set_visible(False)
    fig.tight_layout()
    fig.savefig("sandp.svg")
    plt.plot()


# Normalized stocks
def norm_stock(x):
    fig, ax = plt.subplots(figsize=(7 * cm, 4 * cm))
    for i in range(5):
        ax.plot(x[:, i] + 0.001, lw=1, alpha=0.6,
                c=["C0", "C2", "C1", "C3", "C4"][i])  # label=['^GSPC', '^IXIC', '^NYA', '^N225', '^HSI'][i])
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    fig.savefig("stex.svg", format="svg")


# PCA and TSNE embeddings
def dimred(b1, b2, perp=5, name=""):
    pca = PCA(2)
    pca.fit(np.concatenate((b1, b2)))
    b1_ = pca.transform(b1)
    b2_ = pca.transform(b2)
    sne = TSNE(2, init="pca", learning_rate="auto", perplexity=perp)
    b12 = sne.fit_transform(np.concatenate((b1, b2)))
    m = int(len(b12) / 2)

    fig, ax = plt.subplots(figsize=(7.5 * cm, 7.5 * cm))
    ax.scatter(b1_[:, 0], b1_[:, 1], s=1.5, alpha=0.5)
    ax.scatter(b2_[:, 0], b2_[:, 1], s=1.5, alpha=0.5)
    ax.set_xlabel(r"PCA-x")
    ax.set_ylabel(r"PCA-y")
    fig.savefig("pca_%s.svg" % name)

    fig, ax = plt.subplots(figsize=(7.5 * cm, 7.5 * cm))
    plt.scatter(b12[:m, 0], b12[:m, 1], s=1.5, alpha=0.5)
    plt.scatter(b12[m:, 0], b12[m:, 1], s=1.5, alpha=0.5)
    ax.set_xlabel(r"TSNE-x")
    ax.set_ylabel(r"TSNE-y")
    fig.savefig("tsne_%s.svg" % name)


# Plot one method on one sample
def plot_one(hyperparams, name):
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(hyperparams)
    config = model_config(valid_perc=0.1, test_perc=0.1, hyperparams=hyperparams)
    model, feature_model, _, _, _ = load_net_and_data(config)
    assert "stc_dataset" in config.keys() and "sts_dataset" in config.keys()

    _, _, test_c = load_data(config["stc_dataset"], config["stc_dataset_params"])
    _, _, test_s = load_data(config["sts_dataset"], config["sts_dataset_params"])

    test_c_ = next(iter(test_c))
    test_s_ = next(iter(test_s))
    test_c = Smalldata([test_c_], scale=config["stc_dataset_params"].get("scale", None))
    
    test_s = Smalldata([test_s_], scale=config["sts_dataset_params"].get("scale", None))
    gen = evaluate_style_transfer(
        test_c, test_s, model=model, fmodel=feature_model, **config["st_iter_params"], one=True
    )

    loc = 0

    # Create separate subplots for each sample
    fig, axs = plt.subplots(3, 1, figsize=(7.5 * cm, 15 * cm), sharex=True)

    # Plot the generated sample
    axs[0].plot(gen[loc], lw=1, c="C0", label="Generated Sample", alpha=0.5)
    axs[0].set_ylabel("Generated Sample")
 
    # Plot the content sample if lamb_content hyperparameter is greater than 0
    if hyperparams["lamb_content"] > 0:
        axs[1].plot(test_c.data[0]["x"][loc], lw=1, c="C1", label="Content", alpha=0.5)
        axs[1].set_ylabel("Content")

    # Plot the style sample if lamb_style hyperparameter is greater than 0
    if hyperparams["lamb_style"] > 0:
        axs[2].plot(test_s.data[0]["x"][loc], lw=1, c="C2", label="Style", alpha=0.5)
        axs[2].set_ylabel("Style")

    # Customize the plot appearance
    for ax in axs:
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.set_yticks([-3, 0, 3])

    # Display the plot
    plt.show()

    # Save the plot as an SVG file with a name based on the input parameter 'name'
    fig.savefig('ex_%s.svg' % name, format='svg')
    fig.savefig('svg/ex_%s.svg' % name, format='svg')



# def plot_one(hyperparams, name):
#     seed = 123
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     config = model_config(valid_perc=0.1, test_perc=0.1, hyperparams=hyperparams)
#     model, feature_model, _, _, _ = load_net_and_data(config)
#     assert "stc_dataset" in config.keys() and "sts_dataset" in config.keys()
#     _, _, test_c = load_data(config["stc_dataset"], config["stc_dataset_params"])
#     _, _, test_s = load_data(config["sts_dataset"], config["sts_dataset_params"])

#     test_c_ = next(iter(test_c))
#     test_s_ = next(iter(test_s))
#     test_c = Smalldata([test_c_], scale=config["stc_dataset_params"].get("scale", None))
#     test_s = Smalldata([test_s_], scale=config["sts_dataset_params"].get("scale", None))
#     gen = evaluate_style_transfer(
#         test_c, test_s, model=model, fmodel=feature_model, **config["st_iter_params"], one=True
#     )

#     loc = 0
#     fig, ax = plt.subplots(figsize=(7.5 * cm, 5 * cm))
#     ax.plot(gen[loc], lw=1, c="C0", label="Generated Sample")
#     if hyperparams["lamb_content"] > 0:
#         ax.plot(test_c.data[0]["x"][loc], lw=1, c="C1", label="Content", alpha=0.5)
#     if hyperparams["lamb_style"] > 0:
#         ax.plot(test_s.data[0]["x"][loc], lw=1, c="C2", label="Style", alpha=0.5)
#     ax.spines.right.set_visible(False)
#     ax.spines.top.set_visible(False)
#     ax.set_yticks([-3, 0, 3])
#     plt.show()
#     fig.savefig('ex_%s.svg' % name, format='svg')


# Plot all methods on one sample
def plot_all(methods, name):
    test_c = dict()
    test_s = dict()
    gen = {}
    for k, v in methods.items():
        # Seed
        seed = 123
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        config = model_config(valid_perc=0.1, test_perc=0.1, hyperparams=v)
        model, feature_model, _, _, _ = load_net_and_data(config)
        scale_c = config["stc_dataset_params"].get("scale", "None")
        if scale_c not in test_c.keys():
            assert "stc_dataset" in config.keys() and "sts_dataset" in config.keys()
            _, _, ntest_c = load_data(config["stc_dataset"], config["stc_dataset_params"])
            _, _, ntest_s = load_data(config["sts_dataset"], config["sts_dataset_params"])

            test_c_ = next(iter(ntest_c))
            test_s_ = next(iter(ntest_s))
            test_c[scale_c] = Smalldata([test_c_], scale=config["stc_dataset_params"].get("scale", None))
            test_s[scale_c] = Smalldata([test_s_], scale=config["sts_dataset_params"].get("scale", None))
        gen[k] = evaluate_style_transfer(test_c[scale_c], test_s[scale_c],
                                         model=model, fmodel=feature_model, **config["st_iter_params"], one=True)

    # ns = 3
    # loc = [0, 5, 10]
    ns = 2
    loc = [10, 25]
    fig, ax = plt.subplots(len(methods) + 2, ns, figsize=(15 * cm, 22 * cm), sharex=True, sharey=True)
    for j in range(ns):
        scale = "norm-detrend"
        ax[0, j].plot(test_s[scale].data[0]["x"][loc[j]], lw=1)
        ax[0, 0].set_ylabel("Style", rotation=0, labelpad=30)
        ax[1, j].plot(test_c[scale].data[0]["x"][loc[j]], lw=1)
        ax[1, 0].set_ylabel("Content", rotation=0, labelpad=30)
    for i, (k, v) in enumerate(gen.items(), start=2):
        for j in range(ns):
            ax[i, j].plot(v[loc[j]], lw=1)
        ax[i, 0].set_ylabel(k, rotation=0, labelpad=30)
    for i in range(len(methods) + 2):
        for j in range(ns):
            ax[i, j].spines.left.set_visible(False)
            ax[i, j].spines.top.set_visible(False)
            ax[i, j].yaxis.tick_right()
        [label.set_visible(False) for label in ax[i, 0].get_yticklabels()]
        ax[i, -1].yaxis.set_tick_params(which='both', labelright=True)
        ax[i, -1].set_yticks([-3, 0, 3])
        ax[i, -1].set_yticklabels(["-3", "0", "3"], ha='right')
        for label in ax[i, -1].get_yticklabels():
            label.set_position((label.get_position()[0] + 0.1, label.get_position()[1]))
        # ax[i, 2].yaxis.set_label_position("right")
        # ax[i, 2].set_ylabel("value")
    # for j in range(3):
    #     ax[-1, j].set_xlabel("time")
    plt.show()
    fig.savefig('%s.svg' % name, format='svg')


# Plot all methods on one sample
def plot_all_sp(methods):
    test_c = dict()
    test_s = dict()
    gen = {}
    for k, v in methods.items():
        # Seed
        seed = 123
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        config = model_config(valid_perc=0.1, test_perc=0.1, hyperparams=v)
        model, feature_model, _, _, _ = load_net_and_data(config)
        scale_c = config["stc_dataset_params"].get("scale", "None")
        if scale_c not in test_c.keys():
            assert "stc_dataset" in config.keys() and "sts_dataset" in config.keys()
            ntest_c, _, _ = load_data(config["stc_dataset"], config["stc_dataset_params"])
            ntest_s, _, _ = load_data(config["sts_dataset"], config["sts_dataset_params"])

            test_c_ = next(iter(ntest_c))
            test_s_ = next(iter(ntest_s))
            test_c[scale_c] = Smalldata([test_c_], scale=config["stc_dataset_params"].get("scale", None))
            test_s[scale_c] = Smalldata([test_s_], scale=config["sts_dataset_params"].get("scale", None))
        gen[k] = evaluate_style_transfer(test_c[scale_c], test_s[scale_c],
                                         model=model, fmodel=feature_model, **config["st_iter_params"], one=True)

    # ns = 3
    # loc = [0, 5, 10]
    ns = 2
    loc = [0, 5]
    fig, ax = plt.subplots(len(methods) + 2, ns, figsize=(15 * cm, 22 * cm), sharex=True, sharey=True)
    for j in range(ns):
        scale = "None"
        plot_sp(test_s[scale].data[0]["x"][loc[j]], ax=ax[0, j])
        ax[0, 0].set_ylabel("Style", rotation=0, labelpad=30)
        plot_sp(test_c[scale].data[0]["x"][loc[j]], ax=ax[1, j])
        ax[1, 0].set_ylabel("Content", rotation=0, labelpad=30)
    for i, (k, v) in enumerate(gen.items(), start=2):
        for j in range(ns):
            plot_sp(v[loc[j]], ax=ax[i, j])
        ax[i, 0].set_ylabel(k, rotation=0, labelpad=30)
    for i in range(len(methods) + 2):
        for j in range(ns):
            ax[i, j].spines.left.set_visible(False)
            ax[i, j].spines.top.set_visible(False)
            ax[i, j].yaxis.tick_right()
        [label.set_visible(False) for label in ax[i, 0].get_yticklabels()]
        ax[i, -1].yaxis.set_tick_params(which='both', labelright=True)
        ax[i, -1].set_yticks([])
        for label in ax[i, -1].get_yticklabels():
            label.set_position((label.get_position()[0] + 0.1, label.get_position()[1]))
        # ax[i, 2].yaxis.set_label_position("right")
        # ax[i, 2].set_ylabel("value")
    # for j in range(3):
    #     ax[-1, j].set_xlabel("time")
    plt.show()
    fig.savefig('tim.svg', format='svg')


def print_all(methods, name="sm"):
    prd = dict()
    for k, v in methods.items():
        # Seed
        seed = 123
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        config = model_config(valid_perc=0.1, test_perc=0.1, hyperparams=v)
        model, feature_model, _, _, _ = load_net_and_data(config)
        assert "stc_dataset" in config.keys() and "sts_dataset" in config.keys()
        _, _, test_c = load_data(config["stc_dataset"], config["stc_dataset_params"])
        _, _, test_s = load_data(config["sts_dataset"], config["sts_dataset_params"])

        prd[k] = evaluate_style_transfer(
            test_c, test_s, model=model, fmodel=feature_model, name=name + "_" + k, **config["st_iter_params"]
        )

    fig, ax = plt.subplots(figsize=(10 * cm, 10 * cm))
    for i, (k, v) in enumerate(prd.items()):
        ax.plot(v[1], v[0], label=k + "\$\,\$", alpha=0.8,
                c=["C0", "C1", "C2", "C3", "C4", "C5", "C5", "C5"][i],
                linestyle=["-", "-", "-", "-", "-", "-", (0, (1, 3)), (2, (1, 3))][i]
                )
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel(r'\$\beta\$-Recall')
    ax.set_ylabel(r'\$\alpha\$-Precision')
    # ax.legend(loc="lower left")
    ax.legend()
    fig.savefig("prd_%s.svg" % name, format="svg")
    plt.show()


# example of gaussian process regression
def sketch_gpr():
    gp = GaussianProcessRegressor(RBF(10) + WhiteKernel(0.5), normalize_y=True, n_restarts_optimizer=100)
    X = np.array([[1.6], [2], [2.8], [2.8], [2.8]])
    y = np.array([[0.775], [0.79], [0.72], [0.72], [0.72]])
    X_ = np.linspace(0, 3, 1000).reshape(-1, 1)
    gp.fit(X, y)
    mean, sig = gp.predict(X_, return_std=True)
    mean = mean[:, 0]
    delta = (mean - y.max())
    ie = delta * norm.cdf(delta / sig) + sig * norm.pdf(delta / sig)

    fig, ax = plt.subplots(2, 1, figsize=(15 * cm, 15 * cm), sharex=True)
    ax[0].plot(X_[:, 0], mean, c="C0")
    ax[0].scatter(X[:3, 0], y[:3, 0], c="black", zorder=10)
    ax[0].fill_between(X_[:, 0], (mean + 1.96 * sig), (mean - 1.96 * sig), alpha=0.3, color="C0")
    ax[0].set_ylabel("Predicted MAE")
    ax[0].set_xlabel(r"Parameter \$\vartheta\$")
    ax[0].set_yticks([0.7, 0.75, 0.8])
    ax[0].spines.right.set_visible(False)
    ax[0].spines.top.set_visible(False)
    ax[0].yaxis.set_ticks_position('left')
    ax[0].xaxis.set_ticks_position('bottom')
    ax[1].plot(X_[:, 0], ie, c="C0")
    ax[1].set_ylabel("Expected Improvement")
    ax[1].set_xlabel(r"Parameter \$\vartheta\$")
    ax[1].set_xticks([0, 1, 2, 3])
    ax[1].set_xticklabels([0.001, 0.01, 0.1, 1])
    ax[1].spines.right.set_visible(False)
    ax[1].spines.top.set_visible(False)
    ax[1].yaxis.set_ticks_position('left')
    ax[1].xaxis.set_ticks_position('bottom')
    ax[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.savefig('sketch_gpr.svg', format='svg')
    plt.show()


# extract style transfer sketch from analyze.py
def sketch(target_cpu, target_style_cpu, ts):
    i = 1

    # Represent content and style as fitted polynomial and residuals
    x = np.linspace(-1, 1, target_cpu.shape[1])
    xc = np.polynomial.chebyshev.chebpts1(target_cpu.shape[1])
    pc = np.poly1d(np.polyfit(xc, np.interp(xc, x, target_cpu[i, :, 0]), deg=15))(x)
    ps = np.poly1d(np.polyfit(xc, np.interp(xc, x, target_style_cpu[i, :, 0]), deg=15))(x)

    # Input / Output
    fig, ax = plt.subplots(figsize=(4 * cm, 2.5 * cm))
    ax.plot(x, target_cpu[i], lw=1)
    ax.axis("off")
    fig.savefig('sketch_smooth.svg', format='svg')
    fig, ax = plt.subplots(figsize=(4 * cm, 2.5 * cm))
    ax.plot(x, target_style_cpu[i], lw=1)
    ax.axis("off")
    fig.savefig('sketch_spiky.svg', format='svg')
    fig, ax = plt.subplots(figsize=(4 * cm, 2.5 * cm))
    ax.plot(x, ts[i].detach().cpu(), lw=1)
    ax.axis("off")
    fig.savefig('sketch_transfer.svg', format='svg')

    # Latents
    fig, ax = plt.subplots(figsize=(4 * cm, 2.5 * cm))
    ax.plot(x, pc, lw=1)
    ax.axis("off")
    fig2, ax2 = plt.subplots(figsize=(4 * cm, 2.5 * cm))
    ax2.plot(x, ps, lw=1)
    ax2.axis("off")
    ax2.get_shared_y_axes().join(ax, ax2)
    fig.savefig('sketch_smooth_c.svg', format='svg')
    fig.savefig('sketch_spiky_c.svg', format='svg')
    fig, ax = plt.subplots(figsize=(4 * cm, 2.5 * cm))
    ax.plot(x, target_cpu[i, :, 0] - pc, lw=1)
    ax.axis("off")
    fig2, ax2 = plt.subplots(figsize=(4 * cm, 2.5 * cm))
    ax2.plot(x, target_style_cpu[i, :, 0] - ps, lw=1)
    ax2.axis("off")
    ax2.get_shared_y_axes().join(ax, ax2)
    fig.savefig('sketch_smooth_s.svg', format='svg')
    fig2.savefig('sketch_spiky_s.svg', format='svg')
    plt.show()


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(6*cm, 6*cm))
    x = np.linspace(-1, 1, 100)
    y = x**2
    ax.plot(x, y)
    ax.set_xlabel(r'x \$\rightarrow\$')
    ax.set_ylabel(r'y \$\rightarrow\$')
    fig.savefig('test.svg', format='svg')
    plt.show()