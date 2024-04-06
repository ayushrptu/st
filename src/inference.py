import os
from functools import partial
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from matplotlib import collections as mc
from matplotlib.animation import PillowWriter
import torch
from torchvision.transforms import transforms
import wandb

from config import model_config, NETS_FOLDER
from train_eval import load_net_and_data, load_data
from train_eval import plot_predict
from models import InputOpt
from metrics import Perceptual_Loss, intra_latent_dist
from data import ToyData
from data import data_dict as select_data
from tools import vae_compatible_call, StopForward, Freeze, rescale, InfIter, get_act
from experiment_hyperparams import *
from train_eval import all_eval_
from train_eval import plot_predict
from analyze import feature_analyze
from iterative import iterate
from style_transfer import evaluate_style_transfer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _latent(intra_f, intra_z, intra_content, intra_style, cluster_idxs,
            title="Latent EMSE vs Output Percceptual", perm=None):
    """ perm only tested with loc=out """
    iperm = torch.argsort(perm) if perm is not None else None

    def heat(ax, x, p=None):
        x_ = x[p, :][:, p].detach().cpu() if p is not None else x.detach().cpu()
        ax.imshow(x_, vmin=np.percentile(x_, 5), vmax=np.percentile(x_, 95), cmap="viridis")

    fig, ax = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    heat(ax[0, 0], intra_f, iperm)
    heat(ax[0, 1], intra_z)
    heat(ax[1, 0], intra_style, iperm)
    heat(ax[1, 1], intra_content)
    d1 = torch.sum(cluster_idxs == 0)
    d2 = torch.sum(cluster_idxs == 1)
    d3 = torch.sum(cluster_idxs >= 2)
    for i in range(4):
        for c, d in zip((0, d1, d1 + d2), (d1, d2, d3)):
            ax.reshape(-1)[i].add_patch(patches.Rectangle((c - 0.5, c - 0.5), d, d, ec="red", fc="none", linewidth=3))
    ax[0, 0].set_title("style latent")
    ax[0, 1].set_title("content latent")
    ax[1, 0].set_title("style loss")
    ax[1, 1].set_title("content loss")
    fig.suptitle(title)

    def select(x):
        mask = torch.triu(torch.ones_like(x) == 1, 1)
        return x[mask]

    intra_f = select(intra_f)
    intra_z = select(intra_z)
    intra_content = select(intra_content)
    intra_style = select(intra_style)

    b = cluster_idxs.shape[0]
    cl_con = select(torch.unsqueeze(cluster_idxs, 0).expand([b, b]) + torch.unsqueeze(cluster_idxs, 1).expand([b, b]))
    if perm is not None:
        cluster_idxs = cluster_idxs[perm]
    cl_sty = select(torch.unsqueeze(cluster_idxs, 0).expand([b, b]) + torch.unsqueeze(cluster_idxs, 1).expand([b, b]))
    if cluster_idxs.max() >= 2:
        cmap = ListedColormap(["tab:blue", "tab:brown", "tab:cyan", "orangered", "darkorange", "lawngreen"])
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='smooth - smooth', markerfacecolor=cmap(0 / 5), markersize=5),
            Line2D([0], [0], marker='o', color='w', label='spiky - spiky', markerfacecolor=cmap(2 / 5), markersize=5),
            Line2D([0], [0], marker='o', color='w', label='smooth - spiky', markerfacecolor=cmap(1 / 5), markersize=5),
            Line2D([0], [0], marker='o', color='w', label='gen - smooth', markerfacecolor=cmap(3 / 5), markersize=5),
            Line2D([0], [0], marker='o', color='w', label='gen - spiky', markerfacecolor=cmap(4 / 5), markersize=5),
            Line2D([0], [0], marker='o', color='w', label='gen - gen', markerfacecolor=cmap(5 / 5), markersize=5)
        ]
        s = 1
        alpha = 0.3
    else:
        cmap = get_cmap("tab20")
        if perm is not None:
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='smooth - smooth', markerfacecolor=cmap(0), markersize=5),
                Line2D([0], [0], marker='o', color='w', label='spiky - spiky', markerfacecolor=cmap(1), markersize=5),
                Line2D([0], [0], marker='o', color='w', label='smooth - spiky', markerfacecolor=cmap(0.5),
                       markersize=5),
                Line2D([0], [0], marker='v', color='w', markerfacecolor='none', markeredgecolor="grey",
                       label='smooth - smooth', markersize=5),
                Line2D([0], [0], marker='^', color='w', markerfacecolor='none', markeredgecolor="grey",
                       label='spiky - spiky', markersize=5),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor="grey",
                       label='smooth - spiky', markersize=5)
            ]
        else:
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='smooth - smooth', markerfacecolor=cmap(0), markersize=5),
                Line2D([0], [0], marker='o', color='w', label='spiky - spiky', markerfacecolor=cmap(1), markersize=5),
                Line2D([0], [0], marker='o', color='w', label='smooth - spiky', markerfacecolor=cmap(0.5),
                       markersize=5),
            ]
        s = 5
        alpha = 1

    def scatter(ax, x, y, c, mask):
        ax.scatter(x[mask == 0], y[mask == 0], s=s, cmap=cmap, c=c[mask == 0], alpha=alpha, marker="v")
        ax.scatter(x[mask == 2], y[mask == 2], s=s, cmap=cmap, c=c[mask == 2], alpha=alpha, marker="^")
        mr = (mask != 0) & (mask != 2)
        ax.scatter(x[mr], y[mr], s=s, cmap=cmap, c=c[mr], alpha=alpha)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    if perm is not None:
        scatter(ax[0], intra_style.detach().cpu(), intra_f.detach().cpu(), cl_sty, cl_con)
        scatter(ax[1], intra_content.detach().cpu(), intra_z.detach().cpu(), cl_con, cl_sty)
    else:
        ax[0].scatter(intra_style.detach().cpu(), intra_f.detach().cpu(), s=s, cmap=cmap, c=cl_sty, alpha=alpha)
        ax[1].scatter(intra_content.detach().cpu(), intra_z.detach().cpu(), s=s, cmap=cmap, c=cl_con, alpha=alpha)
    ax[0].set_title("style [r=%.4f]" % torch.corrcoef(torch.stack([intra_f, intra_style]))[0, 1])
    ax[1].set_title("content [r=%.4f]" % torch.corrcoef(torch.stack([intra_z, intra_content]))[0, 1])
    ax[0].set_xlabel("loss")
    ax[1].set_xlabel("loss")
    ax[0].set_ylabel("latent")
    ax[1].set_ylabel("latent")
    ax[0].legend(handles=legend_elements)
    fig.suptitle(title)

    if True:
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
        if perm is not None:
            scatter(ax[0], intra_style.detach().cpu(), intra_content.detach().cpu(), cl_con, cl_sty)
            scatter(ax[1], intra_f.detach().cpu(), intra_z.detach().cpu(), cl_con, cl_sty)
        else:
            ax[0].scatter(intra_style.detach().cpu(), intra_content.detach().cpu(), s=s, cmap=cmap, c=cl_con, alpha=alpha)
            ax[1].scatter(intra_f.detach().cpu(), intra_z.detach().cpu(), s=s, cmap=cmap, c=cl_con, alpha=alpha)
        ax[0].set_title("[r=%.4f]" % torch.corrcoef(torch.stack([intra_content, intra_style]))[0, 1])
        ax[1].set_title("[r=%.4f]" % torch.corrcoef(torch.stack([intra_z, intra_f]))[0, 1])
        ax[0].set_xlabel("style loss")
        ax[1].set_xlabel("style latent")
        ax[0].set_ylabel("content loss")
        ax[1].set_ylabel("content latent")
        ax[0].legend(handles=legend_elements)
        fig.suptitle(title)
    plt.show()


def latent_loss_plot(model, batch, criterion, perm=None, latent_list=None):
    """
    Relationship between latent distance and perceptual loss
    perm: apply permutation to style latents and style labels
    """
    # Get latent vars  todo: get directly from net output later
    if latent_list is None:
        latent = get_act(model, batch, ["fc_mu", "fc_log_std", "fc_f_mu", "fc_f_log_std"])
        mu, log_std, out = latent["fc_mu"], latent["fc_log_std"], latent["out"]
        f_mu, f_log_std = latent["fc_f_mu"], latent["fc_f_log_std"]
        if perm is not None:
            f_mu = f_mu[perm]
            f_log_std = f_log_std[perm]
        std = torch.exp(log_std)
        f_std = torch.exp(f_log_std)
    else:
        f_mu, f_std, mu, std = latent_list

    # Sort by cluster
    cluster_idxs = batch["didx"]
    c_idxs, id = torch.sort(cluster_idxs, stable=True)
    perm_ = torch.argsort(id)[perm[id]] if perm is not None else None

    intra_f, intra_z = intra_latent_dist(f_mu, f_std, mu, std)
    intra_f, intra_z = intra_f[id, :][:, id], intra_z[id, :][:, id]

    with torch.no_grad():
        outputs = torch.zeros_like(batch["x"])
        ths_batch = batch["x"][cluster_idxs < 2]
        ths_mask = batch["mask"][cluster_idxs < 2]
        outputs[cluster_idxs < 2] = model(ths_batch.to(device), src_mask=(ths_mask == 1).to(device),
                                          style_perm=perm)[0].cpu()
        outputs[cluster_idxs >= 2] = batch["x"][cluster_idxs >= 2]

    criterion._get_activations(outputs=outputs, labels=batch["y"].to(device))
    intra_content, intra_style = criterion.intra_content_style_loss(loc="out", style_perm=perm)
    intra_content, intra_style = intra_content[id, :][:, id], intra_style[id, :][:, id]
    _latent(intra_f, intra_z, intra_content, intra_style, c_idxs, "Latent EMSE vs Output Perceptual", perm_)

    intra_content, intra_style = criterion.intra_content_style_loss(loc="in", style_perm=perm)
    intra_content, intra_style = intra_content[id, :][:, id], intra_style[id, :][:, id]
    _latent(intra_f, intra_z, intra_content, intra_style, c_idxs, "Latent EMSE vs Label Perceptual", perm_)


def latent_plot(model, batch):
    latent = get_act(model, batch, ["fc_mu", "fc_log_std"])
    mu, log_std, out = latent["fc_mu"], latent["fc_log_std"], latent["out"]
    std = torch.exp(log_std)

    fig, ax = plt.subplots(4, 1, figsize=(16, 10))
    ax[0].plot(batch["x"][0])
    ax[1].plot(np.array(mu[0].detach().cpu()))
    ax[2].plot(np.array(std[0].detach().cpu()))
    ax[3].plot(np.array(out[0].detach().cpu()))
    ax[0].set_title("input")
    ax[1].set_title("latent means")
    ax[2].set_title("latent stds")
    ax[3].set_title("output sample")
    plt.show()


def dis_latent_plot(model, batch):
    latent = get_act(model, batch, ["fc_mu", "fc_log_std", "fc_f_mu", "fc_f_log_std"])
    mu, log_std, out = latent["fc_mu"], latent["fc_log_std"], latent["out"]
    f_mu, f_log_std = latent["fc_f_mu"], latent["fc_f_log_std"]
    std = torch.exp(log_std)
    f_std = torch.exp(f_log_std)

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    spec = fig.add_gridspec(4, 2)
    ax = [fig.add_subplot(spec[0, :]), fig.add_subplot(spec[1, 0]), fig.add_subplot(spec[1, 1]),
          fig.add_subplot(spec[2, 0]), fig.add_subplot(spec[2, 1]), fig.add_subplot(spec[3, :])]

    ax[0].plot(batch["x"][0])
    ax[1].plot(np.array(mu[0].detach().cpu()))
    ax[2].plot(np.array(f_mu[0].T.detach().cpu()))
    ax[3].plot(np.array(std[0].detach().cpu()))
    ax[4].plot(np.array(f_std[0].T.detach().cpu()))
    ax[5].plot(np.array(out[0].detach().cpu()))
    ax[0].set_title("input")
    ax[1].set_title("latent content means")
    ax[2].set_title("latent style means")
    ax[3].set_title("latent content stds")
    ax[4].set_title("latent style stds")
    ax[5].set_title("output sample")
    plt.show()


def analyze(load, data=None, perm=None, use_train=False):
    # file_name = "toy_data_noise_0_5.csv"
    file_name = "toy_data.csv"

    if data is None:
        # Load from preset
        hyperparams = VAE_DIST
        # CNN.update({"out_conv_channels": 32 * 2 ** 4})
        hyperparams.update(
            # dataset="CSV", file_name=file_name,
            dataset="CSVs", file_names=["toy_data.csv", "toy_data_noise_0_5.csv"], csv_weights=[1, 1],
            load_artifact=load, transform="masking",  # cloze_perc=0, mask_rand_none_split=(1.0, 0.0, 0.0),
            d_latent=4, d_latent_style=8, pe_after_latent=True,
            style_loss="mean_std", content_hooks=["conv0.0"], style_hooks=["conv1.0"], last_hook="conv1.0",
            feature_model="CNN", feature_model_params=CNN, feature_load_artifact="bert-tiny-3mvy1523:v19"
        )

        config = model_config(valid_perc=0.1, test_perc=0.1, hyperparams=hyperparams)
        model, fmodel, train, val, test = load_net_and_data(config)
        model.eval()
        criterion = Perceptual_Loss(fmodel, **config["training_params"]["criterion_params"])
        batch = next(iter(train if use_train else val))
    else:
        model = data["model"]
        batch = data["batch"]
        criterion = data["criterion"]
    if perm is None:
        perm = torch.randperm(batch["x"].shape[0])

    # hyperparams.update(load_artifact="bert-tiny-24moxam4:v0")
    # config = model_config(valid_perc=0.1, test_perc=0.1, hyperparams=hyperparams)
    # model2, _, _, _, _ = load_net_and_data(config)
    print("everything loaded")

    # eval
    # res1 = all_eval_(model, fmodel, val, ["MSE", "DIS-perceptual"], config["training_params"]["criterion_params"])
    # res2 = all_eval_(model, fmodel, val, ["MSE", "DIS-perceptual"], config["training_params"]["criterion_params"], style_perm=perm)
    # print(res1)
    # print(res2)

    # explore activations
    # latent_plot(model, batch)
    # dis_latent_plot(model, batch)
    latent_loss_plot(model, batch, criterion)
    latent_loss_plot(model, batch, criterion, perm=perm)


def style_transfer(model, batch1, batch2, criterion=None, transform=None):
    """
    put style of batch ts2 in batch ts1
    if criterion is not None, analyze wrt that
    """
    latent2 = get_act(model, batch2, ["fc_mu", "fc_log_std", "fc_f_mu", "fc_f_log_std"], last_hook="fc_log_std")
    f_mu2 = latent2["fc_f_mu"]
    f_std2 = torch.exp(latent2["fc_f_log_std"])
    mu2 = latent2["fc_mu"]
    std2 = torch.exp(latent2["fc_log_std"])

    with torch.no_grad():
        out_transfer = model(batch1["x"].to(device), src_mask=batch1["mask"] == 1, style=[f_mu2, f_std2])[0]

    latent1 = get_act(model, batch1, ["fc_mu", "fc_log_std", "fc_f_mu", "fc_f_log_std"])
    f_mu1 = latent1["fc_f_mu"]
    f_std1 = torch.exp(latent1["fc_f_log_std"])
    mu1 = latent1["fc_mu"]
    std1 = torch.exp(latent1["fc_log_std"])
    out = latent1["out"]

    in1 = batch1["x"]
    in2 = batch2["x"]

    # if criterion is not None:
    #     out_transfer = feature_analyze([criterion.model, in1, in2, out_transfer], transform=transform, cl=1, sl=100, num_iters=20, lr=250000, fps=2)
    #     # out_transfer = feature_analyze([criterion.model, in1, in2, 2 * torch.rand(out_transfer.shape) - 1], transform=transform, cl=1, sl=100, num_iters=500, lr=0.1, fps=20)
    #     criterion = None

    if transform is not None:
        in1 = transform(in1)
        in2 = transform(in2)
        out = transform(out)
        out_transfer = transform(out_transfer)

    # 1
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    spec = fig.add_gridspec(4, 4)
    ax = [fig.add_subplot(spec[0, 0:2]), fig.add_subplot(spec[0, 2:4]),
          fig.add_subplot(spec[1, 0:2]), fig.add_subplot(spec[1, 2]), fig.add_subplot(spec[1, 3]),
          fig.add_subplot(spec[2, 0:2]), fig.add_subplot(spec[2, 2]), fig.add_subplot(spec[2, 3]),
          fig.add_subplot(spec[3, 0:2]), fig.add_subplot(spec[3, 2:4])]
    ax[3].get_shared_y_axes().join(ax[3], ax[4])
    ax[6].get_shared_y_axes().join(ax[6], ax[7])
    ax[8].get_shared_y_axes().join(ax[8], ax[9])
    ax[4].set_yticklabels([])
    ax[7].set_yticklabels([])
    ax[9].set_yticklabels([])

    i = 0
    ax[0].plot(in1[i].detach().cpu())
    ax[1].plot(in2[i].detach().cpu())
    ax[2].plot(np.array(mu1[i].detach().cpu()))
    ax[3].plot(np.array(f_mu1[i].T.detach().cpu()))
    ax[4].plot(np.array(f_mu2[i].T.detach().cpu()))
    ax[5].plot(np.array(std1[i].detach().cpu()))
    ax[6].plot(np.array(f_std1[i].T.detach().cpu()))
    ax[7].plot(np.array(f_std2[i].T.detach().cpu()))
    ax[8].plot(np.array(out[i].detach().cpu()))
    ax[8].set_prop_cycle(None)
    ax[8].plot(np.array(in1[i].detach().cpu()), alpha=0.3, lw=1)
    ax[9].plot(np.array(out_transfer[i].detach().cpu()))
    ax[9].set_prop_cycle(None)
    ax[9].plot(np.array(in2[i].detach().cpu()), alpha=0.3, lw=1)
    ax[0].set_title("input 1 (content)")
    ax[1].set_title("input 2 (style)")
    ax[2].set_title("latent content means [1]")
    ax[3].set_title("latent content means [1]")
    ax[4].set_title("latent style means [2]")
    ax[5].set_title("latent content stds [1]")
    ax[6].set_title("latent content stds [1]")
    ax[7].set_title("latent style stds [2]")
    ax[8].set_title("output sample: content 1, style 1")
    ax[9].set_title("output sample: content 1, style 2")
    plt.show()

    if criterion is not None:
        comb_batch = {"x": torch.cat([batch1["x"], batch2["x"], out_transfer.cpu()]),
                      "didx": torch.repeat_interleave(torch.tensor([0, 1, 3]), batch1["x"].shape[0]),
                      "mask": (torch.zeros_like(batch1["mask"]) == 1).repeat([3, 1]),
                      "y": torch.cat([batch1["y"], batch2["y"], out_transfer.cpu()])}
        latent_list = [torch.cat([f_mu1, f_mu2, f_mu2]), torch.cat([f_std1, f_std2, f_std2]),
                       torch.cat([mu1, mu2, mu1]), torch.cat([std1, std2, std1])]
        latent_loss_plot(model, comb_batch, criterion, latent_list=latent_list)

    return [out, out_transfer]


def plt_st_many(batch1, batch2, out1, out_st1, out2, out_st2):
    # More
    fig, ax = plt.subplots(6, 8, figsize=(16, 10))
    for cax in ax.reshape(-1):
        cax.set_xticks([])
        cax.set_yticks([])
        cax.spines['top'].set_visible(False)
        cax.spines['right'].set_visible(False)
        cax.spines['bottom'].set_visible(False)
        cax.spines['left'].set_visible(False)
    ax[0, 0].get_shared_y_axes().join(ax[0, 0], *ax.reshape(-1)[1:])
    ax[0, 0].get_shared_x_axes().join(ax[0, 0], *ax.reshape(-1)[1:])
    for i in range(8):
        ax[0, i].plot(batch1["x"][i].detach().cpu(), lw=1)
        ax[1, i].plot(batch2["x"][i].detach().cpu(), lw=1)
        ax[2, i].plot(np.array(out1[i].detach().cpu()), lw=1)
        ax[3, i].plot(np.array(out_st1[i].detach().cpu()), lw=1)
        ax[4, i].plot(np.array(out_st2[i].detach().cpu()), lw=1)
        ax[5, i].plot(np.array(out2[i].detach().cpu()), lw=1)
    ax[0, 0].set_ylabel("input 1", rotation=0, labelpad=50)
    ax[1, 0].set_ylabel("input 2", rotation=0, labelpad=50)
    ax[2, 0].set_ylabel("output sample:\n content 1, style 1", rotation=0, labelpad=50)
    ax[3, 0].set_ylabel("output sample:\n content 1, style 2", rotation=0, labelpad=50)
    ax[4, 0].set_ylabel("output sample:\n content 2, style 1", rotation=0, labelpad=50)
    ax[5, 0].set_ylabel("output sample:\n content 2, style 2", rotation=0, labelpad=50)
    plt.show()


def plt_in_out(model, data1, data2, k=5):
    plot_predict(model, data1, title="Reconstruction", id="reconstruct_1", k=k)
    plot_predict(model, data2, title="Reconstruction", id="reconstruct_2", k=k)


def inference(load, type=0, task="show", data=None, use_train=False):
    if data is None:

        mode = "finance"
        if mode == "finance":
            file_name = "stock_vol_interp.csv"
            # Load from preset
            hyperparams = VAE_DIST
            hyperparams.update(
                layers=2, d_model=256, d_latent=8, d_latent_style=16, batch_size=8,
                pe_after_latent=True, seq_len=128,
                dataset="LabeledCSV", file_name=file_name, scale="norm-all-detrend", force_same_class=True,
                relevant_cols=['^GSPC', '^IXIC', '^NYA', '^N225', '^HSI'], select_class=-1,
                # dataset="CSVs", file_names=["toy_data.csv", "toy_data_noise_0_5.csv"], csv_weights=[1, 1],
                load_artifact=load, transform="masking", cloze_perc=0, mask_rand_none_split=(1.0, 0.0, 0.0),
                style_loss="mean_std", content_hooks=["conv0.0"], style_hooks=["conv1.0"], last_hook="conv1.0",
                feature_model="CNN", feature_model_params=CNN_f, feature_load_artifact="bert-tiny-3nbtffo6:latest"
            )
            config = model_config(valid_perc=0.1, test_perc=0.1, hyperparams=hyperparams)
            model, fmodel, train, val, test = load_net_and_data(config)
            model.eval()
            model.to(device)
            criterion = Perceptual_Loss(fmodel, **config["training_params"]["criterion_params"])

            # def transform(ts):
            #     return torch.cumprod(torch.exp(ts.detach() / 30), 1)

            transform = None

            # Load 2nd dataset
            config["dataset_params"]["select_class"] = 1
            # config["dataset_params"]["test_perc"] = 0
            # config["dataset_params"]["valid_perc"] = 0
            # val2 = val
            train2, val2, test2 = select_data[config["dataset"]](**config["dataset_params"]).dataloader()

        else:
            data_sets = ["toy_data_noise_0_5.csv", "toy_data.csv"]
            file_content_id, file_style_id = type, 1 - type
            file_name_content = data_sets[file_content_id]
            file_name_style = data_sets[file_style_id]

            # Load from preset
            hyperparams = VAE_DIST
            hyperparams.update(
                dataset="CSV", file_name=file_name_content,
                # dataset="CSVs", file_names=["toy_data.csv", "toy_data_noise_0_5.csv"], csv_weights=[1, 1],
                load_artifact=load, transform="masking", cloze_perc=0, mask_rand_none_split=(1.0, 0.0, 0.0),
                d_latent=8, d_latent_style=16, pe_after_latent=True,
                style_loss="mean_std", content_hooks=["conv0.0"], style_hooks=["conv1.0"], last_hook="conv1.0",
                feature_model="CNN", feature_model_params=CNN, feature_load_artifact="bert-tiny-3mvy1523:v19"
            )
            config = model_config(valid_perc=0.1, test_perc=0.1, hyperparams=hyperparams)
            model, fmodel, train, val, test = load_net_and_data(config)
            model.eval()
            model.to(device)
            criterion = Perceptual_Loss(fmodel, **config["training_params"]["criterion_params"])

            # Load 2nd dataset
            head, _ = os.path.split(config["dataset_params"]["file_path"])
            config["dataset_params"]["file_path"] = os.path.join(head, file_name_style)
            train2, val2, test2 = select_data[config["dataset"]](**config["dataset_params"]).dataloader()
            transform = None

        print("everything loaded")

        iter1 = iter(train if use_train else val)
        iter2 = iter(train2 if use_train else val2)
        batch1 = next(iter1)
        batch1_2 = next(iter1)
        batch2 = next(iter2)
        batch2_2 = next(iter2)

    else:
        print(data)
        model = data["model"]
        batch1 = data["batch1"]
        batch2 = data["batch2"]
        criterion = data["criterion"]

    # inference-time tests, e.g. precision-recall / predictive utility
    if task == "test":
        evaluate_style_transfer(train, train2, model=None, fmodel=fmodel, cl=1, sl=10, num_iters=500, lr=0.01)
    # inference-time analysis
    elif task == "analyze":
        outs1 = style_transfer(model, batch1, batch2, criterion=criterion, transform=transform)
        outs2 = style_transfer(model, batch2, batch1, criterion=criterion, transform=transform)
    # inference-time style transfer
    else:
        outs2 = style_transfer(model, batch2, batch1, transform=transform)
        outs1 = style_transfer(model, batch1, batch2, transform=transform)
        plt_st_many(batch2, batch1, *outs2, *outs1)
        try:
            plt_in_out(model, train, train2, k=5)
            model.eval()
        except:
            pass


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # torch.use_deterministic_algorithms(True)

    # load = "bert-tiny-3o69sao4:latest"  # sp100
    # load = "bert-tiny-24rqhooh:latest"  # sp100+ (high lam_kl)
    # load = "bert-tiny-1hykyw37:latest"  #sp100++ (higher lam_kl)
    # load = "bert-tiny-1x09t121:latest"  # sm1000 !no pretrain
    # load = "bert-tiny-bzx1bm2l:latest"   # sm-sp100
    # load = "bert-tiny-1y6iju0e:latest"   # sm-sp100+
    # load = "bert-tiny-2qn6ctrq:latest"   # sm-sp100++
    # load = "bert-tiny-1i7fd0rt:latest"    # sp-sm100 !no pretrain

    # load = "bert-tiny-2gfpmdif:latest"  # sp100+ dim1
    # load = "bert-tiny-6hzfk5zb:latest"   # sp100+ pe

    # load = "bert-tiny-33tw8pg4:latest"  # dis 8 4
    # load = "bert-tiny-23t6oyn1:latest"  # dis 4 1
    # load = "bert-tiny-2lh4ee4a:latest"  # dis 8 4 pe
    # load = "bert-tiny-3g6lua9f:latest"  # dis 8 4 both
    # load = "bert-tiny-1ngj7q1e:latest"  # dis 3/1
    # load = "bert-tiny-2v9eyp3o:latest"  # dis 1/1

    # load = "bert-tiny-l5kz6kgs:latest"  # [shuffle] - good encoding

    # load = "bert-tiny-1rixjisi:latest"  # no con, no dis
    # load = "bert-tiny-24m3cxjt:latest"  # no con
    # load = "bert-tiny-2cywgvm7:latest"  # no sty
    # load = "bert-tiny-26h9nikf:latest"  # no pe

    # load = "bert-tiny-tah34wnt:latest"  # half-1
    # load = "bert-tiny-l8c6is8t:latest"  # half-10
    # load = "bert-tiny-2b4hd0nj:latest"  # cur-5
    # load = "bert-tiny-11kcov00:v50"  # cur-7
    # load = "bert-tiny-1b3u6b0y:latest"  # cur-10
    # load = "bert-tiny-5j95a2od:latest"   # full
    # load = "bert-tiny-24hbuhmz:latest"   # cur-1

    # load = "bert-tiny-x61tufv1:latest"  # finance 5 pretrain
    # load = "bert-tiny-2yvjczg7:latest"  # + dis
    # load = "bert-tiny-2qfqe230:latest"  # - kl
    load = "bert-tiny-258niv9l:latest"  # half
    # load = "bert-tiny-3h7yi7r3:latest"  # full

    # inference(load, task="show")
    # inference(load, task="analyze")
    inference(load, task="test")
    # analyze(load)
