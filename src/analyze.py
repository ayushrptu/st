import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.animation import PillowWriter
import torch

from config import model_config, NETS_FOLDER
from train_eval import load_net_and_data, load_data
from train_eval import plot_predict
from models import InputOpt
from metrics import Perceptual_Loss
from data import ToyData
from data import data_dict as select_data
from experiment_hyperparams import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot(train, val, model, k=1, transform=None):
    plot_predict(model, train, title="Reconstruction (train)", id="reconstruct_train", k=k, transform=transform)
    plot_predict(model, val, title="Reconstruction (val)", id="reconstruct_val", k=k, transform=transform)
    plt.show()


def test_style(data, model):
    gp = ToyData(n=1, noise=0)

    # Likelihood to be created by ToyData
    lk_pre = []
    lk_post = []
    model.eval()
    with torch.no_grad():
        for batch in data:
            labels = batch["y"]
            for i in range(labels.shape[0]):
                lk_pre += [gp.likelihood(labels[i].numpy())]
            inputs = batch["x"].to(device)
            mask = (batch["mask"] > 0).to(device)
            outputs = model(inputs, src_mask=mask)
            if type(outputs) == tuple:
                outputs = outputs[0]
            outputs = outputs.to("cpu")
            for i in range(outputs.shape[0]):
                lk_post += [gp.likelihood(outputs[i].numpy())]
    model.train()
    lk_dif = [post - pre for (pre, post) in zip(lk_pre, lk_post)]

    print("mean likelihood (data): %.2f" % (sum(lk_pre) / len(lk_pre)))
    print("mean likelihood (output): %.2f" % (sum(lk_post) / len(lk_post)))
    plt.hist([lk_pre, lk_post], label=["ground truth", "reconstruction"])
    plt.title("Log-Likelihood")
    plt.legend()
    plt.figure()
    plt.hist(lk_dif)
    plt.title("Shift in Log-Likelihood")
    plt.show()


def kernels(net, weight="context.weight"):
    """ Visualize kernels """
    nm_paras = dict(net.named_parameters())
    try:
        kernel = nm_paras[weight].detach().cpu()
    except KeyError:
        print("Can't find convolutional kernel")
        return

    order = np.argsort([f.std() for f in kernel])[::-1].copy()
    print(order)
    global ORDER
    ORDER = order
    kernel = kernel[ORDER]

    fig, ax = plt.subplots(kernel.shape[0], 1, figsize=(8, 10))
    mi = np.percentile(kernel, 2)
    ma = np.percentile(kernel, 98)
    for i in range(kernel.shape[0]):
        ax[i].set_axis_off()
        ax[i].imshow(kernel[i], cmap="RdBu_r", vmin=mi, vmax=ma)
    # fig.tight_layout()

    # fig, ax = plt.subplots(3, 2)
    # cmap = cm.get_cmap("viridis")
    # for i in range(3):
    #     for j, n in enumerate(["weight", "bias"]):
    #         try:
    #             data = np.array(nm_paras["fc%d.%s" % (i + 1, n)].detach().cpu())
    #         except KeyError:
    #             data = np.array(nm_paras["fc%d.0.%s" % (i + 1, n)].detach().cpu())
    #         if len(data.shape) < 2:
    #             data = data.reshape(-1, 1)
    #         data = data[np.argsort(data.std(axis=1))[::-1], :]
    #         ax[i, j].hist(data.T, bins=25, stacked=True,
    #                       color=cmap(np.log(np.linspace(np.exp(0), np.exp(1), data.T.shape[1]))))
    #         ax[i, j].set_title("layer %d - %s" % (i + 1, n))
    #         ax[i, j].set_yscale("log")
    # fig.tight_layout()
    plt.show()


def explain_filter(net, batch, feature="context"):
    """ show filter activations on a batch """

    # Hook to get feature activations
    feats = {}
    def hook_func(m, inp, op):
        feats[feature] = op.cpu().detach()
    dict(net.named_modules())[feature].register_forward_hook(hook_func)

    input = batch["x"]
    with torch.no_grad():
        output = net(input.to(device), src_mask=(batch["mask"] == 1).to(device))[0].cpu()
    filter = feats[feature].permute(0, 2, 1)[:, :, ORDER]

    rows, _, cols = np.minimum(10, filter.shape)
    fig, ax = plt.subplots(rows, cols, figsize=(15, 8))
    for i in range(rows):
        for j in range(cols):
            x = np.arange(filter.shape[1])
            y = input[i, :, 0].numpy()
            # interpolate
            y = np.interp(x, np.linspace(0, filter.shape[1], input.shape[1]), y)
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            lc = mc.LineCollection(np.concatenate([points[:-1], points[1:]], axis=1), cmap="RdYlGn")
            lc.set_array(filter[i, :, j].numpy())
            lc.set_linewidth(2)
            ax[i, j].add_collection(lc)
            ax[i, j].set_xlim(x.min(), x.max())
            ax[i, j].set_ylim(y.min() - 0.1, y.max() + 0.1)
            ax[i, j].set_yticks([])
            ax[i, j].set_xticks([])
            ax[i, j].spines['top'].set_visible(False)
            ax[i, j].spines['right'].set_visible(False)
            ax[i, j].spines['bottom'].set_visible(False)
            ax[i, j].spines['left'].set_visible(False)
        ax[i, 0].set_ylabel("Sample %d" % i, rotation=0, labelpad=25)
    for j in range(cols):
        ax[rows-1, j].set_xlabel("Feature %d" % j)
    plt.show()


def model_analyze():
    # load = 'bert_tiny-278c61b6:latest'  # BERT
    # load = "bert_tiny-3scam1wa:latest"  # BERT+conv+post
    # load = "bert_tiny-6so6gzli:latest"  # BERT+conv+pre
    # load = "bert_tiny-2j9dta9v:latest"  # BERT+post
    # load = "bert_tiny-2d2wyqmb:latest"  # BERT+pre
    # load = "bert_tiny-17h1imoo:latest"  # BERT+conv
    # load = "bert_tiny-iuwd4vf1:v1"  # BERT_Style
    # load = "bert_tiny-2lv9oql4:v4"  # BERT_Style+conv+post (noise)
    # load = "bert_tiny-1hf99dqm:v2"  # BERT_Style+conv+pre
    # load = "bert_tiny-2hj3hs5o:v4"  # biLSTM_Style
    # load = "bilstm-34x037q4:v4"  # BiLSTM_Style + conv+post
    # load = "bilstm-2lx9xz9m:v4"  # BiLSTM_Style + noise
    # load = "bert-tiny-oslhj1pe:v14"   # BERT_VAE l0.1 64
    # load = "bert-tiny-2n7c39zr:v11"   # BERT_VAE l0.1 2
    # load = "bert-tiny-3vpm3f2b:v19"     # BERT_VAE l0.1 best smooth
    # load = "bert-tiny-yern2jlv:v19"     # BERT_VAE l0.1 best noisy
    # load = "bert-tiny-pl1100jn:v19"       # CNN 16 sm
    # load = "bert-tiny-3mvy1523:v19"       # CNN 32 sm
    # load = "bert-tiny-t3tuli6y:v19"  # l2 sm
    # load = "bert-tiny-h5v9mq2b:v19"   # CNNAE sp
    # load = "bert-tiny-26q6dtky:v19"   # CNNAE sm
    # load = "bert-tiny-3o69sao4:latest"  # sp100
    # load = "bert-tiny-24rqhooh:latest"  # sp100+ (high lam_kl)
    # load = "bert-tiny-1hykyw37:latest"  #sp100++ (higher lam_kl)
    # load = "bert-tiny-1x09t121:latest"  # sm1000 !no pretrain
    # load = "bert-tiny-bzx1bm2l:latest"   # sm-sp100
    # load = "bert-tiny-1y6iju0e:latest"   # sm-sp100+
    # load = "bert-tiny-2qn6ctrq:latest"   # sm-sp100++
    # load = "bert-tiny-1i7fd0rt:latest"    # sp-sm100 !no pretrain

    load = "bert-tiny-d326sa6o:latest"  # CNN finance 5

    # Load from preset
    mode = "finance"
    if mode == "finance":
        hyperparams = CNN
        hyperparams.update(out_conv_channels=16 * 2 ** 8, dropout=0.1)
        hyperparams.update(dataset="LabeledCSV", file_name="high_low_vol_interp.csv", scale=30,
                           relevant_cols=['^GSPC', '^IXIC', '^NYA', '^N225', '^HSI'])
    else:
        hyperparams = VAE
        # CNN.update({"out_conv_channels": 32 * 2 ** 4})
        file_name = "toy_data_noise_0_5.csv"
        # file_name = "toy_data.csv"
        hyperparams.update(dataset="CSV", file_name=file_name)

    hyperparams.update(load_artifact=load, transform="masking", cloze_perc=0,
                       mask_rand_none_split=(1.0, 0.0, 0.0))

    config = model_config(valid_perc=0.1, test_perc=0.1, hyperparams=hyperparams)
    model, fmodel, train, val, test = load_net_and_data(config)
    print("everything loaded")

    def stock(ts):
        return np.cumprod(np.exp(ts / 30))

    # style transfer
    plot(train, val, model, k=3, transform=stock)
    # test_style(train, model)
    # test_style(val, model)

    # explore first convolution
    # batch = next(iter(val))
    # feature = "conv3.0"
    # kernels(model, feature + ".weight")
    # explain_filter(model, batch, feature)
    # visualize_filter(net)


def feature_analyze(loaded=None, ch=0, sh=1, cl=1, sl=100, ls="ms", lr=0.1, num_iters=500, transform=None, fps=36):
    """
    optimize input wrt perceptual loss on given features

    :param loaded: [model, batch_content, batch_style, init]
    :param ch: content hook layer
    :param sh: style   hook layer
    :param cl: content weight
    :param sl: style   weight
    :param ls: loss, "ms" (mean_std) or "gm" (gram)
    """
    if loaded is None:
        file_content_id, file_style_id = 0, 1
        style_extra = file_content_id != file_style_id

        # Load from preset
        mode = "finance"
        if mode == "finance":
            # load = "bert-tiny-r1krxsf8:latest"  # CNN finance norm
            # load = "bert-tiny-2m2twpsy:latest"  # CNN finance norm-detrend
            # load = "bert-tiny-1bvz1r8z:latest"  # CNN finance norm-all
            load = "bert-tiny-3nbtffo6:latest"  # CNN finance norm-all-detrend

            hyperparams = CNN
            hyperparams.update(out_conv_channels=16 * 2 ** 8, dropout=0.2, seq_len=256)
            hyperparams.update(dataset="LabeledCSV", file_name="stock_vol_interp.csv", scale="norm-all-detrend", force_same_class=True,
                               select_class=file_content_id,
                               relevant_cols=['^GSPC', '^IXIC', '^NYA', '^N225', '^HSI'])

            transform = None
            # def transform(ts):
            #     return np.cumprod(np.exp(ts / 30))
        else:
            # load = "bert-tiny-3ncza3ua:v19"  # CNN sm 32
            # load = "bert-tiny-pl1100jn:v19"  # CNN sm 16
            load = "bert-tiny-3mvy1523:v19"  # CNN sp 32

            data = ["toy_data.csv", "toy_data_noise_0_5.csv"]
            file_name_content = data[file_content_id]

            # Load from preset
            hyperparams = CNN
            CNN.update({"out_conv_channels": 32 * 2 ** 4})
            hyperparams.update(dataset="CSV", file_name=file_name_content)
            transform = None

        hyperparams.update(load_artifact=load, transform="masking", cloze_perc=0, batch_size=4,
                           mask_rand_none_split=(1.0, 0.0, 0.0))
        config = model_config(valid_perc=0.1, test_perc=0.1, hyperparams=hyperparams)
        model, _, train, val, test = load_net_and_data(config)

        # Load 2nd dataset
        if style_extra:
            if mode == "finance":
                config["dataset_params"]["select_class"] = file_style_id
                config["dataset_params"]["test_perc"] = 0
                config["dataset_params"]["valid_perc"] = 0
                # val2 = val
                val2 = select_data[config["dataset"]](**config["dataset_params"]).dataloader()[0]

            else:
                file_name_style = data[file_style_id]
                head, _ = os.path.split(config["dataset_params"]["file_path"])
                config["dataset_params"]["file_path"] = os.path.join(head, file_name_style)
                train2, val2, test2 = select_data[config["dataset"]](**config["dataset_params"]).dataloader()

        print("everything loaded")

        batch = next(iter(val))
        target_cpu = batch["y"]
        target = batch["y"].to(device)
        if style_extra:
            batch = next(iter(val2))
            target_style_cpu = batch["y"]
            target_style = batch["y"].to(device)
        else:
            target_style_cpu, target_style = None, None
        net = InputOpt(*target.shape).to(device)
        name = "../img/%s[%d-%d]_lc%d_ls%d_%s.gif" % (mode, ch, sh, cl, sl, ls)
    else:
        model, batch, batch_style, init = loaded
        target_cpu = batch.detach().cpu()
        target = batch.detach().to(device)
        target_style_cpu = batch_style.detach().cpu()
        target_style = batch_style.detach().to(device)
        style_extra = True
        net = InputOpt(*target.shape, init=init).to(device)
        name = "test.gif"

    lss = {"ms": "mean_std", "gm": "gram"}[ls]
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    criterion = Perceptual_Loss(model, content_hooks=["conv%d.0" % ch], style_hooks=["conv%d.0" % sh], last_hook=["conv4.0"],
                                lamb_content=cl, lamb_style=sl, style_loss=lss)

    fig, ax = plt.subplots(2, 2)
    if style_extra:
        lim = [min(target_cpu.min(), target_style_cpu.min()) - 0.3, max(target_cpu.max(), target_style_cpu.max()) + 0.3]
    else:
        lim = [target_cpu.min() - 0.3, target_cpu.max() + 0.3]
    if transform is not None:
        lim = [transform(lim[0]), transform(lim[1])]
        lim = [lim[0] - 2 * (lim[1] - lim[0]), lim[1] + 2 * (lim[1] - lim[0])]
    writer = PillowWriter(fps=fps)

    with writer.saving(fig, name, 100):
        for frame in range(num_iters):
            if frame % 5 == 0:
                print(frame)
            optimizer.zero_grad()
            ts = net()
            loss = criterion(ts, target, style_labels=target_style)["perceptual"]
            loss.backward()
            optimizer.step()
            if frame > 100:
                scheduler.step()

            for i, cax in enumerate(ax.reshape(-1)):
                cax.cla()
                cax.set_axis_off()
                cax.set_ylim(*lim)
                tg = target_cpu[i, :, 0].detach()
                tgs = target_style_cpu[i, :, 0].detach()
                t = ts[i, :, 0].cpu().detach()
                if transform is not None:
                    tg = transform(tg)
                    tgs = transform(tgs)
                    t = transform(t)
                cax.plot(tg, alpha=0.5)
                if style_extra:
                    cax.plot(tgs, alpha=0.5)
                cax.plot(t)
            writer.grab_frame()
    return net().detach().cpu()


if __name__ == "__main__":
    np.random.seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model_analyze()
    feature_analyze()
