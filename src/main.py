import random

import numpy as np
import torch

import wandb
from config import model_config
from plot import print_all, plot_all, plot_all_sp, plot_one
from train_eval import train_model
from experiment_hyperparams import *

PROJECT = "st_feature_one_channel"


def local(**hyperparams):
    config = model_config(valid_perc=0.1, test_perc=0.1, hyperparams=hyperparams)
    train_model(config)


def online(**hyperparams):
    run = wandb.init(project=PROJECT, reinit=True)
    config = model_config(valid_perc=0.1, test_perc=0.1, hyperparams=hyperparams, mode="wandb")
    wandb.config.update(config)
    train_model(config)
    run.finish()


def train():
    wandb.init()
    config = model_config(valid_perc=0.1, test_perc=0.1, hyperparams=wandb.config, mode="wandb")
    wandb.config.update(config)
    train_model(config)


def sweep(count=10, name="BERT", metric="val_MSE", lr=True, lr_st=False, **hyperparams):
    """
    Sweep where hyperparams can have lists too
    """
    # Seed
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    sweep_config = {
        "name": name,
        "metric": {
            "name": metric,
            "goal": "minimize"
        },
        "method": "bayes",
        "parameters": {}
    }
    if lr:
        sweep_config["parameters"].update(**{
            "lr": {
                "max": 0.01,
                "min": 0.000001,
                "distribution": "log_uniform_values"
            },
        })
    if lr_st:
        sweep_config["parameters"].update(**{
            "st_lr": {
                "max": 0.01,
                "min": 0.000001,
                "distribution": "log_uniform_values"
            },
        })

    for k, v in hyperparams.items():
        if k not in ["relevant_cols", "mask_rand_none_split"] and type(v) == tuple:
            if k in ["lr_decay", "st_lr_decay"]:
                sweep_config["parameters"][k] = {"max": v[0], "min": v[1]}
            elif k in ["lamb_kl", "lamb_content", "lamb_style", "st_lr", "lr"]:
                sweep_config["parameters"][k] = {"max": v[0], "min": v[1], "distribution": "log_uniform_values"}
            else:
                sweep_config["parameters"][k] = {"values": v}
        else:
            sweep_config["parameters"][k] = {"value": v}
    sweep_id = wandb.sweep(sweep_config, project=PROJECT)
    wandb.agent(sweep_id, function=train, count=count)


def existing(count=5):
    # sweep_id = "197wqm9t"  # cloze=10
    # sweep_id = "hi7vp4we"  # cloze=1
    # sweep_id = "g553hqw1"  # debug
    sweep_id = "n4crjkph"  # fine-tune
    wandb.agent(sweep_id, function=train, count=count, project=PROJECT)


if __name__ == "__main__":

    tasks = ["gp-train-iter"]
    methods_sm2sp = {}
    methods_sp2sm = {}

    # -------------------------------------------------------------------

    if "gp-pretrain-cnn" in tasks:
        # loss =
        cnn_pretrain_gp = dict(
            **GP_feat_train, epochs=10, lr_delay=0, lr_full=1, lr_decay=0.7, loss="MSE"  # 0.7**9 = 0.04
        )
        # online(**cnn_pretrain_gp)
        cnn_pretrain_gp.update(
            count=10, metric="val_MSE",
            out_conv_channels=tuple([x * 2 ** 8 for x in [2, 4, 8, 16]]),
            layers=(2, 3, 4)
        )
        sweep(**cnn_pretrain_gp)

    # I-DAE
    if "gp-train-iter" in tasks:
        idea = dict(**I_DAE, **LAMBS, st_lr_full=200)
        idea_r = dict(**I_DAE_r, **LAMBS)
        # online(**copy_update(idea, st_lr_decay=0.9566, st_lr=0.00995, **TEST))
        # online(**copy_update(idea, st_lr_decay=0.9872, st_lr=0.1804, **TEST))  # r
        # online(**copy_update(idea, st_lr_decay=0.8339, st_lr=0.7826, num_iters=5, **TEST))
        # online(**copy_update(idea, st_lr_decay=0.8892, st_lr=1.1340, num_iters=10, **TEST))
        # online(**copy_update(idea, st_lr_decay=0.7963, st_lr=0.8520, num_iters=25, **TEST))
        # online(**copy_update(idea, st_lr_decay=0.9752, st_lr=0.6131, num_iters=100, **TEST))
        # methods_sm2sp["I-DEA"] = copy_update(idea, st_lr_decay=0.9821, st_lr=0.8975, num_iters=500)
        # methods_sp2sm["I-DEA"] = copy_update(idea_r, st_lr_decay=0.9872, st_lr=0.1804, num_iters=500)

        # plot_one = lambda x, name=None: online(**x)

        # plot_one(copy_update(idea, st_lr_decay=1, st_lr=0.001, num_iters=100, **TEST), name="base")
        # plot_one(copy_update(idea, st_lr_decay=0.9998, st_lr=0.0091, num_iters=300, **TEST,
        #                      content_hooks=["conv0.0"], style_hooks=["conv0.0"], lamb_style=3), name="s_dp")
        # plot_one(copy_update(idea, st_lr_decay=0.9998, st_lr=0.0091, num_iters=500, **TEST,
        #                      content_hooks=["conv2.0"], style_hooks=["conv1.0"], last_hook="conv2.0"), name="c_dp")
        # plot_one(copy_update(idea, st_lr_decay=0.9998, st_lr=0.0091, num_iters=400, style_loss="gram", **TEST), name="gram")
        # plot_one(copy_update(idea, st_lr_decay=0.9821, st_lr=0.8975, num_iters=500, lamb_content=0, **TEST), name="con")
        # plot_one(copy_update(idea_r, st_lr_decay=0.9821, st_lr=0.8975, num_iters=500, lamb_content=0, **TEST), name="sty")
        # idea.update(
        #     count=20, metric="stval_perceptual", lr=False, st_lr=(5, 0.00001), st_lr_decay=(1.0, 0.7),
        # )
        # sweep(**copy_update(idea, num_iters=5), name="i5")
        # sweep(**copy_update(idea, num_iters=10), name="i10")
        # sweep(**copy_update(idea, num_iters=25), name="i25")
        # sweep(**copy_update(idea, num_iters=100), name="i100")
        sweep(**copy_update(idea, num_iters=500), name="i500")

    # I-HC
    if "gp-train-iter_hc" in tasks:
        ihc = dict(**I_HC, **LAMBS, num_iters=500, st_lr_full=200)
        ihc_r = dict(**I_HC_r, **LAMBS, num_iters=500, st_lr_full=200)
        methods_sm2sp["I-HC"] = copy_update(ihc, st_lr_decay=0.9021, st_lr=0.0100)
        methods_sp2sm["I-HC"] = copy_update(ihc_r, st_lr_decay=0.9130, st_lr=0.0096)
        # local(**copy_update(ihc, st_lr_decay=0.9021, st_lr=0.0100, **TEST))
        ihc.update(
            count=20, metric="stval_fin", lr=False, st_lr=(10, 0.00001), st_lr_decay=(1.0, 0.8),
        )
        # sweep(**ihc)

    # M-FT
    if "gp-train-mft" in tasks:
        mft = dict(
            **M_FT1, **LAMBS, epochs=50, lr_full=20, lr_decay=0.8715,
            layers=4, dim_feedforward=128, d_model=32
        )
        mft_r = dict(
            ** M_FT1_r, **LAMBS, epochs=50, lr_full=20, lr_decay=0.8715,
            layers=4, dim_feedforward=1024, d_model=128
        )
        mft.update(
            count=10, metric="val_perceptual", lr=True, lr_decay=(1.0, 0.8), epochs=(10, 20, 50),
            layers=(2, 3, 4), dim_feedforward=(128, 256, 512), d_model=(32, 64, 128),
        )
        mft_r.update(
            count=10, metric="val_perceptual", lr=True, lr_decay=(1.0, 0.8), epochs=(10, 20, 50),
            layers=(2, 3, 4), dim_feedforward=(128, 256, 512), d_model=(32, 64, 128),
        )
        # sweep(**mft)
        # sweep(**mft_r)
        mft2 = dict(
            **M_FT2_, **LAMBS, epochs=10, lr_delay=1, lr_full=4, lr_decay=0.8919, lr=0.0067,
            layers=4, dim_feedforward=128, d_model=32, load_artifact="bert-tiny-6gepmcnn:latest"
        )
        mft2_r = dict(
            ** M_FT2_r_, **LAMBS, epochs=10, lr_delay=1, lr_full=4, lr_decay=0.9900, lr=0.0094,
            layers=4, dim_feedforward=1024, d_model=128, load_artifact="bert-tiny-5xq5agqi:latest"
        )
        methods_sm2sp["M-FT"] = copy_update(mft2, skip_train=True, load_artifact="bert-tiny-9bcw1o1w:latest")
        methods_sp2sm["M-FT"] = copy_update(mft2_r, skip_train=True, load_artifact="bert-tiny-garigxeq:latest")
        # local(**copy_update(mft2, skip_train=True, load_artifact="bert-tiny-9bcw1o1w:latest", **TEST))
        # local(**copy_update(mft2, skip_train=True, load_artifact="bert-tiny-garigxeq:latest", **TEST)) # r
        # online(**copy_update(mft2, skip_train=True, load_artifact="bert-tiny-x2wwiycw:latest", **TEST))  # ptft
        mft2.update(
            count=10, metric="stval_perceptual", lr=True, lr_decay=(1.0, 0.8),
        )
        mft2_r.update(
            count=10, metric="stval_perceptual", lr=True, lr_decay=(1.0, 0.8),
        )
        # sweep(**mft2)
        # sweep(**mft2_r)

    # MA-CNN
    if "gp-train-macnn" in tasks:
        macnn = dict(
            **MA_CNN1, **LAMBS, epochs=50, lr_full=20, lr_decay=0.8128, lr=0.00002, lamb_kl=0.0007,
            layers=2, d_model=32, d_latent=1, d_latent_style=16, kernel_size=25
        )
        macnn.update(
            count=20, metric="val_perceptual", lr=True, lr_decay=(1.0, 0.8), epochs=(10, 20, 50),
            lamb_kl=(10, 0.00001), layers=(2, 3, 4), d_model=(32, 64, 128), d_latent=(1, 2, 4, 8),
            d_latent_style=(1, 2, 4, 8, 16), kernel_size=(5, 10, 25)
        )
        # sweep(**macnn)

        macnn2 = dict(
            **MA_CNN2, **LAMBS, epochs=10, lr_delay=0, lr_full=4, lr_decay=0.8801, lr=0.00055, lamb_kl=0.0004,
            layers=2, d_model=32, d_latent=1, d_latent_style=16, kernel_size=25,
            load_artifact="bert-tiny-3i3gc14s:latest"
        )
        # local(**copy_update(macnn2, skip_train=True, load_artifact="bert-tiny-gvmqsb3k:latest", **TEST))
        methods_sm2sp["MA-CNN"] = copy_update(macnn2, skip_train=True, load_artifact="bert-tiny-gvmqsb3k:latest")
        methods_sp2sm["MA-CNN"] = copy_update(macnn2, skip_train=True, load_artifact="bert-tiny-gvmqsb3k:latest")
        macnn2.update(
            count=10, metric="stval_perceptual", lr=True, lr_decay=(1.0, 0.8), lamb_kl=(10, 0.00001),
        )
        # sweep(**macnn2)

    # MA-LSTM
    if "gp-train-malstm" in tasks:
        malstm = dict(
            **MA_LSTM1, **LAMBS, epochs=50, lr_full=20, lr_decay=0.8128, lr=0.00002, lamb_kl=0.0007,
            layers=2, d_model=64, d_latent=2, d_latent_style=1
        )
        # local(**malstm)
        malstm.update(
            count=20, metric="val_perceptual", lr=True, lr_decay=(1.0, 0.8), epochs=(10, 20, 50),
            lamb_kl=(10, 0.00001),
            layers=(2, 3, 4), d_model=(32, 64, 128), d_latent=(1, 2, 4, 8), d_latent_style=(1, 2, 4, 8, 16)
        )
        # sweep(**malstm)

        malstm2 = dict(
            **MA_LSTM2, **LAMBS, epochs=10, lr_delay=0, lr_full=4, lr_decay=0.8031, lr=0.00001, lamb_kl=0.00008,
            layers=2, d_model=64, d_latent=2, d_latent_style=1,
            load_artifact="bert-tiny-popsc0q0:latest"
        )
        # online(**copy_update(malstm2, skip_train=True, load_artifact="bert-tiny-hfiazkt1:latest", **TEST)) # pe
        # local(**copy_update(malstm2, skip_train=True, load_artifact="bert-tiny-3sug5u1h:latest", **TEST))
        methods_sm2sp["MA-LSTM"] = copy_update(malstm2, skip_train=True, load_artifact="bert-tiny-3sug5u1h:latest")
        methods_sp2sm["MA-LSTM"] = copy_update(malstm2, skip_train=True, load_artifact="bert-tiny-3sug5u1h:latest")
        malstm2.update(
            count=10, metric="stval_perceptual", lr=True, lr_decay=(1.0, 0.8), lamb_kl=(10, 0.00001),
        )
        # sweep(**malstm2)

    # MA-T
    if "gp-train-mat" in tasks:
        mat = dict(
            **MA_T1, **LAMBS, epochs=50, lr_full=20, lr_decay=0.8845, lr=0.00036, lamb_kl=0.00001,
            layers=2, dim_feedforward=512, d_model=128, d_latent=8, d_latent_style=4
        )
        mat.update(
            count=10, metric="val_perceptual", lr=True, lr_decay=(1.0, 0.8), epochs=(10, 20, 50), lamb_kl=(10, 0.00001),
            layers=(2, 3, 4), dim_feedforward=(128, 256, 512), d_model=(32, 64, 128),
            d_latent=(1, 2, 4, 8), d_latent_style=(1, 2, 4, 8, 16)
        )
        # sweep(**mat)

        mat2 = dict(
            **MA_T2, **LAMBS, epochs=10, lr_delay=0, lr_full=4, lr_decay=1, lr=0.001, lamb_kl=0.0004,
            layers=2, dim_feedforward=512, d_model=128, d_latent=8, d_latent_style=4,
            load_artifact="bert-tiny-bxk4p1j0:latest"
        )
        methods_sm2sp["MA-T"] = copy_update(mat2, skip_train=True, load_artifact="bert-tiny-kearl7xi:latest")
        methods_sm2sp["MA-T-10"] = copy_update(mat2, skip_train=True, num_iters=10, st_lr_decay=0.9678, st_lr=0.1050, load_artifact="bert-tiny-kearl7xi:latest")
        methods_sm2sp["MA-T-25"] = copy_update(mat2, skip_train=True, num_iters=25, st_lr_decay=0.9514, st_lr=0.1143, load_artifact="bert-tiny-kearl7xi:latest")
        methods_sp2sm["MA-T"] = copy_update(mat2, skip_train=True, load_artifact="bert-tiny-kearl7xi:latest")
        methods_sp2sm["MA-T-10"] = copy_update(mat2, skip_train=True, num_iters=10, st_lr_decay=0.9042, st_lr=0.0651, load_artifact="bert-tiny-kearl7xi:latest")
        methods_sp2sm["MA-T-25"] = copy_update(mat2, skip_train=True, num_iters=25, st_lr_decay=0.8766, st_lr=0.0346, load_artifact="bert-tiny-kearl7xi:latest")
        # local(**copy_update(mat2, skip_train=True, load_artifact="bert-tiny-kearl7xi:latest", **TEST))
        # online(**copy_update(mat2, skip_train=True, num_iters=5, st_lr_decay=0.9981, st_lr=0.1354, load_artifact="bert-tiny-kearl7xi:latest", **TEST))
        # local(**copy_update(mat2, skip_train=True, num_iters=10, st_lr_decay=0.9678, st_lr=0.1050, load_artifact="bert-tiny-kearl7xi:latest", **TEST))
        # local(**copy_update(mat2, skip_train=True, num_iters=25, st_lr_decay=0.9514, st_lr=0.1143, load_artifact="bert-tiny-kearl7xi:latest", **TEST))
        # online(**copy_update(mat2, skip_train=True, num_iters=100, st_lr_decay=0.9705, st_lr=0.4917, load_artifact="bert-tiny-kearl7xi:latest", **TEST))
        # online(**copy_update(mat2, skip_train=True, num_iters=500, st_lr_decay=0.9919, st_lr=0.2625, load_artifact="bert-tiny-kearl7xi:latest", **TEST))
        # online(**copy_update(mat2, skip_train=True, num_iters=10, st_lr_decay=0.9042, st_lr=0.0651, load_artifact="bert-tiny-kearl7xi:latest", **TEST))  # r
        # online(**copy_update(mat2, skip_train=True, num_iters=25, st_lr_decay=0.8766, st_lr=0.0346, load_artifact="bert-tiny-kearl7xi:latest", **TEST))  # r
        # online(**copy_update(mat2, skip_train=True, num_iters=5, st_lr_decay=0.9061, st_lr=0.1298, load_artifact="bert-tiny-kearl7xi:latest", **TEST))
        # online(**copy_update(mat2, skip_train=True, num_iters=10, st_lr_decay=0.8555, st_lr=0.1310, load_artifact="bert-tiny-kearl7xi:latest", **TEST))
        # online(**copy_update(mat2, skip_train=True, num_iters=25, st_lr_decay=0.9851, st_lr=0.0696, load_artifact="bert-tiny-kearl7xi:latest", **TEST))
        # online(**copy_update(mat2, skip_train=True, num_iters=100, st_lr_decay=0.9652, st_lr=0.3142, load_artifact="bert-tiny-kearl7xi:latest", **TEST))
        # online(**copy_update(mat2, skip_train=True, num_iters=500, st_lr_decay=0.9948, st_lr=0.2115, load_artifact="bert-tiny-kearl7xi:latest", **TEST))

        # online(**copy_update(mat2, lamb_style=1, **TEST))
        # online(**copy_update(mat2, lamb_style=100, **TEST))

        # del mat2["load_artifact"]
        # online(**copy_update(mat2, epochs=10, **TEST))  # no_pt

        # online(**copy_update(mat2, skip_train=True, load_artifact="bert-tiny-5xkf35uc:latest", **TEST)) # ft2
        mat2_i = copy_update(mat2,
            count=20, metric="stval_perceptual", skip_train=True, lr=False, st_lr_decay=(1.0, 0.8),
            st_lr=(2, 0.001), st_lr_full=0, num_iters=10, load_artifact="bert-tiny-kearl7xi:latest", name="t10"
        )
        mat2.update(
            count=10, metric="stval_perceptual", lr=True, lr_decay=(1.0, 0.8), lamb_kl=(10, 0.00001),
        )
        # sweep(**mat2)
        # sweep(**copy_update(mat2_i, num_iters=5, name="t5"))
        # sweep(**copy_update(mat2_i, num_iters=10, name="t10"))
        # sweep(**copy_update(mat2_i, num_iters=25, name="t25"))
        # sweep(**copy_update(mat2_i, num_iters=100, name="t100"))
        # sweep(**copy_update(mat2_i, num_iters=500, name="t500"))

        # mat3 = dict(
        #     **MA_T3, **LAMBS, epochs=1, lr=0.00055, lamb_kl=0.0004, load_artifact="bert-tiny-kearl7xi:latest",
        #     layers=2, dim_feedforward=512, d_model=128, d_latent=8, d_latent_style=4
        # )
        # # local(**mat3)
        # mat3.update(
        #     count=10, metric="stval_perceptual", lr=True, lamb_kl=(0.1, 0.00001),
        # )
        # # sweep(**mat3)

    # plot_all(methods_sm2sp)
    # print_all(methods_sm2sp, name="sm")
    # plot_all(methods_sp2sm)
    # print_all(methods_sp2sm, name="sp")

    # -------------------------------------------------------------------

    # tasks = ["fin-train-iter", "fin-train-iter_hc", "fin-train-mft", "fin-train-macnn", "fin-train-malstm", "fin-train-mat"]
    methods_lo2hi = {}

    # -------------------------------------------------------------------

    if "fin-pretrain-cnn" in tasks:
        # loss =
        fin_cnn = dict(
            **FIN_feat_train, epochs=50, lr_delay=0, lr_full=5, lr_decay=0.9, loss="MSE",
            layers=4, out_conv_channels=16 * 2 ** 8, dropout=0.2
        )
        online(**fin_cnn)
        fin_cnn.update(
            count=10, metric="val_MSE", lr=True, lr_decay=(1.0, 0.8),
            out_conv_channels=tuple([x * 2 ** 8 for x in [2, 4, 8, 16]]), layers=(2, 3, 4),
        )
        # sweep(**fin_cnn)

    # I-DAE
    if "fin-train-iter" in tasks:
        idea = dict(**F_I_DAE, **LAMBS)
        # online(**copy_update(idea, st_lr_decay=0.9655, st_lr=0.3618, num_iters=500, **TEST))
        methods_lo2hi["I-DEA"] = copy_update(idea, st_lr_decay=0.9655, st_lr=0.3618, num_iters=500)

        # plot_one = lambda x, name=None: online(**x)

        # plot_one(copy_update(idea, st_lr_decay=0.9655, st_lr=0.3618, num_iters=500, **TEST), name="base")
        # plot_one(copy_update(idea, st_lr_decay=0.9655, st_lr=0.3618, num_iters=500, lamb_style=1, **TEST), name="base")
        # plot_one(copy_update(idea, st_lr_decay=0.9655, st_lr=0.3618, num_iters=500, lamb_style=100, **TEST), name="base")

        idea.update(
            count=20, metric="stval_perceptual", lr=False, st_lr=(5, 0.00001), st_lr_decay=(1.0, 0.7),
        )
        # sweep(**copy_update(idea, num_iters=500), name="i500")

    # I-HC
    if "fin-train-iter_hc" in tasks:
        ihc = dict(**F_I_HC, **LAMBS, num_iters=500)
        methods_lo2hi["I-HC"] = copy_update(ihc, st_lr_decay=0.9959, st_lr=0.01598)
        # online(**copy_update(ihc, st_lr_decay=0.9981, st_lr=0.1269, **TEST))
        ihc.update(
            count=20, metric="stval_fin", lr=False, st_lr=(10, 0.00001), st_lr_decay=(1.0, 0.7),
        )
        # sweep(**ihc)

    # M-FT
    if "fin-train-mft" in tasks:
        mft = dict(
            **F_M_FT1, **LAMBS, epochs=50, lr_full=20, lr_decay=0.8715,
            layers=3, dim_feedforward=512, d_model=128
        )
        mft.update(
            count=10, metric="val_perceptual", lr=True, lr_decay=(1.0, 0.8), epochs=(10, 20, 50),
            layers=(2, 3, 4), dim_feedforward=(128, 256, 512), d_model=(32, 64, 128),
        )
        # sweep(**mft)
        mft2 = dict(
            **F_M_FT2_, **LAMBS, epochs=10, lr_delay=1, lr_full=4, lr_decay=0.8919, lr=0.0067,
            layers=3, dim_feedforward=512, d_model=128, load_artifact="bert-tiny-0b9r91zm:latest"
        )
        methods_lo2hi["M-FT"] = copy_update(mft2, skip_train=True, load_artifact="bert-tiny-yssgxd9r:latest")
        # online(**copy_update(mft2, skip_train=True, load_artifact="bert-tiny-yssgxd9r:latest", **TEST))
        mft2.update(
            count=10, metric="stval_perceptual", lr=True, lr_decay=(1.0, 0.8),
        )
        # sweep(**mft2)

    # MA-CNN
    if "fin-train-macnn" in tasks:
        macnn = dict(
            **F_MA_CNN1, **LAMBS, epochs=150, lr_delay=1, lr_full=20,
            layers=3, d_model=64, d_latent=4, d_latent_style=16, kernel_size=25
        )
        macnn2 = dict(
            **F_MA_CNN2, **LAMBS, epochs=10, lr_delay=0, lr_full=4, lr_decay=0.9578, lr=0.00166,
            layers=3, d_model=64, d_latent=4, d_latent_style=16, kernel_size=25,
            load_artifact="bert-tiny-b63gasb2:latest"
        )
        macnn3 = copy_update(macnn2, load_artifact="bert-tiny-nxhg68ch:latest", shuffle_style="c1", lamb_kl=0.00001)

        # online(**copy_update(macnn3, skip_train=True, load_artifact="bert-tiny-3123ae81:latest", **TEST))
        methods_lo2hi["MA-CNN"] = copy_update(macnn3, skip_train=True, load_artifact="bert-tiny-3123ae81:latest", **TEST)

        macnn.update(
            count=10, metric="val_perceptual", lr=True, lr_decay=(1.0, 0.8), epochs=(10, 20, 50, 150),
            lamb_kl=(10, 0.00001),
            layers=(2, 3, 4), d_model=(32, 64, 128), kernel_size=(5, 10, 25),
            d_latent=(1, 2, 4, 8), d_latent_style=(1, 2, 4, 8, 16)
        )
        macnn2.update(
            count=5, metric="stval_perceptual", lr=(0.1, 0.000001), lr_decay=(1.0, 0.8), lamb_kl=(10, 0.00001),
        )
        macnn3.update(
            count=5, metric="stval_perceptual", lr=True, lr_decay=(1.0, 0.8), lamb_kl=(10, 0.00001),
        )
        # sweep(**macnn)
        # sweep(**macnn2)
        # sweep(**macnn3)

    # MA-LSTM
    if "fin-train-malstm" in tasks:
        malstm = dict(
            **F_MA_LSTM1, **LAMBS, epochs=150, lr_full=20, lr_decay=0.8845, lr=0.00036, lamb_kl=0.00001,
            layers=3, d_model=128, d_latent=8, d_latent_style=16
        )
        malstm2 = dict(
            **F_MA_LSTM2, **LAMBS, epochs=10, lr_delay=0, lr_full=4, lr_decay=0.9578, lr=0.00166, lamb_kl=0.0003296,
            layers=3, d_model=128, d_latent=8, d_latent_style=16,
            load_artifact="bert-tiny-8opk3kzi:latest"
        )
        malstm3 = copy_update(malstm2, load_artifact="bert-tiny-g4znhart:latest", shuffle_style="c1", lamb_kl=0.00001)

        # online(**copy_update(malstm3, skip_train=True, load_artifact="bert-tiny-n22hmqqy:latest", **TEST))
        methods_lo2hi["MA-LSTM"] = copy_update(malstm3, skip_train=True, load_artifact="bert-tiny-n22hmqqy:latest", **TEST)

        malstm.update(
            count=10, metric="val_perceptual", lr=True, lr_decay=(1.0, 0.8), epochs=(10, 20, 50, 150),
            lamb_kl=(10, 0.00001),
            layers=(2, 3, 4), d_model=(32, 64, 128),
            d_latent=(1, 2, 4, 8), d_latent_style=(1, 2, 4, 8, 16)
        )
        malstm2.update(
            count=5, metric="stval_perceptual", lr=True, lr_decay=(1.0, 0.8), lamb_kl=(10, 0.00001),
        )
        malstm3.update(
            count=5, metric="stval_perceptual", lr=True, lr_decay=(1.0, 0.8), lamb_kl=(10, 0.00001),
        )
        # sweep(**malstm)
        # sweep(**malstm2)
        # sweep(**malstm3)

    # MA-T
    if "fin-train-mat" in tasks:
        mat = dict(
            **F_MA_T1, **LAMBS, epochs=150, lr_full=20, lr_decay=0.8845, lr=0.00036, lamb_kl=0.00001,
            layers=3, dim_feedforward=512, d_model=128, d_latent=8, d_latent_style=16
        )
        mat2 = dict(
            **F_MA_T2, **LAMBS, epochs=10, lr_delay=0, lr_full=4, lr_decay=0.9578, lr=0.00166, lamb_kl=0.0003296,
            layers=3, dim_feedforward=512, d_model=128, d_latent=8, d_latent_style=16,
            load_artifact="bert-tiny-96ppmjd2:latest"
        )
        mat3 = copy_update(mat2, load_artifact="bert-tiny-nayq7sak:latest", shuffle_style="c1", lamb_kl=0.00001)
        mat3_i = copy_update(mat3, load_artifact="bert-tiny-0b5ysgeg:latest", skip_train=True, st_lr_full=0)

        # online(**copy_update(mat2, skip_train=True, load_artifact="bert-tiny-nayq7sak:latest", **TEST))  # noft
        # online(**copy_update(mat3, skip_train=True, load_artifact="bert-tiny-0b5ysgeg:latest", **TEST))
        # online(**copy_update(mat3_i, num_iters=10, st_lr_decay=0.9919, st_lr=0.07694, **TEST))
        # online(**copy_update(mat3_i, num_iters=25, st_lr_decay=0.8007, st_lr=0.09674, **TEST))

        # online(**copy_update(mat3, lr=0.0001058, lr_decay=0.9952, lamb_style=1, epochs=20, **TEST)) lamb
        # online(**copy_update(mat3, lr=0.0001058, lr_decay=0.9952, lamb_style=100, epochs=20, **TEST))

        methods_lo2hi["MA-T"] = copy_update(mat3, skip_train=True, load_artifact="bert-tiny-0b5ysgeg:latest", **TEST)
        methods_lo2hi["MA-T-10"] = copy_update(mat3_i, num_iters=10, st_lr_decay=0.9919, st_lr=0.07694, **TEST)
        methods_lo2hi["MA-T-25"] = copy_update(mat3_i, num_iters=25, st_lr_decay=0.8007, st_lr=0.09674, **TEST)

        # online(**copy_update(mat3_i, skip_train=True, load_artifact="bert-tiny-v2fgdo1b:latest", num_iters=1000,
        #                      lamb_content=0.0012, lamb_style=198, **TEST))  #pMAE
        # online(**copy_update(mat3, lamb_content=1, lamb_style=100, **TEST))
        # online(**copy_update(mat3, lamb_content=1, lamb_style=10, **TEST))
        # online(**copy_update(mat3, lamb_content=1, lamb_style=1, **TEST))
        # online(**copy_update(mat3, lamb_content=0.0001, lamb_style=1000, **TEST))

        mat.update(
            count=10, metric="val_perceptual", lr=True, lr_decay=(1.0, 0.8), epochs=(10, 20, 50, 150),
            lamb_kl=(10, 0.00001),
            layers=(2, 3, 4), dim_feedforward=(128, 256, 512), d_model=(32, 64, 128),
            d_latent=(1, 2, 4, 8), d_latent_style=(1, 2, 4, 8, 16)
        )
        mat2.update(
            count=10, metric="stval_perceptual", lr=True, lr_decay=(1.0, 0.8), lamb_kl=(10, 0.00001),
        )
        mat3_ = copy_update(mat3,
            count=10, metric="stval_p-MAE", lr=True, lr_decay=(1.0, 0.8), lamb_kl=(10, 0.00001),
            lamb_content=(10, 0.00001), lamb_style=(1000, 0.001)
        )
        mat3.update(
            count=10, metric="stval_perceptual", lr=True, lr_decay=(1.0, 0.8), lamb_kl=(10, 0.00001),
        )
        mat3_i.update(
            count=20, metric="stval_perceptual", lr=False, st_lr_decay=(1.0, 0.6), st_lr=(2, 0.0001),
        )
        # sweep(**mat)
        # sweep(**mat2)
        # sweep(**mat3)
        # sweep(**mat3_)
        # sweep(**mat3_i, num_iters=10, name="t10")
        # sweep(**mat3_i, num_iters=25, name="t10")

    # plot_all(methods_lo2hi)
    # print_all(methods_lo2hi, name="fin")
    for k, h in methods_lo2hi.items():
        if k == "I-HC":
            methods_lo2hi[k] = copy_update(h, st_content_params=FIN_S_n)
        else:
            methods_lo2hi[k] = copy_update(h, st_content_params=FIN_S)
    # plot_all(methods_lo2hi, name="sim")
    # print_all(methods_lo2hi, name="sim")
    for k, h in methods_lo2hi.items():
        print("\n\n" + k + "\n\n\n")
        online(**copy_update(h, **TEST))


    # -------------------------------------------------------------------

    # tasks = ["tim-pretrain-cnn"]
    # tasks = ["tim-train-iter", "tim-train-mat"]
    methods_m2f = {}

    # -------------------------------------------------------------------

    if "tim-pretrain-cnn" in tasks:
        # loss =
        tim_cnn = dict(
            **TIM_feat_train, epochs=0.1, lr_delay=0, lr_full=1, loss="MSE",
            layers=3, out_conv_channels=8 * 2 ** 8, dropout=0.2
        )
        online(**tim_cnn, lr=0.0001)
        tim_cnn.update(
            count=10, metric="val_MSE", lr=True,
            # out_conv_channels=tuple([x * 2 ** 8 for x in [2, 4, 8, 16]]), layers=(2, 3, 4),
        )
        # sweep(**tim_cnn)

    # I-DAE
    if "tim-train-iter" in tasks:
        idea = dict(**T_I_DAE, **LAMBS)
        # online(**copy_update(idea, st_lr_decay=0.9655, st_lr=0.3618, num_iters=500, **TEST))
        methods_m2f["I-DEA"] = copy_update(idea, st_lr_decay=0.9655, st_lr=0.3618, num_iters=500, **TEST)
        idea.update(
            count=20, metric="stval_perceptual", lr=False, st_lr=(5, 0.00001), st_lr_decay=(1.0, 0.7),
        )
        # sweep(**copy_update(idea, num_iters=500), name="i500")

    # MA-T
    if "tim-train-mat" in tasks:
        mat2 = dict(
            **T_MA_T2, **LAMBS, epochs=1, lr_delay=0, lr=0.00001, lamb_kl=0.0003296,
            layers=3, dim_feedforward=512, d_model=128, d_latent=8, d_latent_style=16
        )
        # mat3 = copy_update(mat2, shuffle_style="c1", lamb_kl=0.00001)

        # online(**copy_update(mat2, **TEST))
        methods_m2f["MA-T"] = copy_update(mat2, load_artifact="bert-tiny-lu6eb0uk:latest", skip_train=True)
        # online(**copy_update(mat3, skip_train=True, load_artifact="bert-tiny-0b5ysgeg:latest", **TEST))

    # plot_all_sp(methods_m2f)

    # -------------------------------------------------------------------

    tasks = ["gpm-pretrain-cnn", "gpm-train-iter", "gpm-train-mat"]

    methods_sm2sp = {}
    methods_sp2sm = {}

    # -------------------------------------------------------------------

    if "gpm-pretrain-cnn" in tasks:
        # loss =
        cnn_pretrain_gp = dict(
            **GP_feat_train, epochs=10, lr_delay=0, lr_full=1, lr_decay=0.7, loss="MSE"  # 0.7**9 = 0.04
        )
        # online(**cnn_pretrain_gp)
        cnn_pretrain_gp.update(
            count=10, metric="val_MSE",
            out_conv_channels=tuple([x * 2 ** 8 for x in [2, 4, 8, 16]]),
            layers=(2, 3, 4)
        )
        # sweep(**cnn_pretrain_gp)

    # I-DAE
    if "gpm-train-iter" in tasks:
        idea = dict(**I_DAE, **LAMBS, st_lr_full=200)
        idea_r = dict(**I_DAE_r, **LAMBS)
        # online(**copy_update(idea, st_lr_decay=0.9752, st_lr=0.6131, num_iters=500, **TEST))
        methods_sm2sp["I-DEA"] = copy_update(idea, st_lr_decay=0.9821, st_lr=0.8975, num_iters=500)
        methods_sp2sm["I-DEA"] = copy_update(idea_r, st_lr_decay=0.9872, st_lr=0.1804, num_iters=500)

        # plot_one = lambda x, name=None: online(**x)

        # plot_one(copy_update(idea, st_lr_decay=0.9998, st_lr=0.0091, num_iters=100, **TEST), name="base")
        # plot_one(copy_update(idea, st_lr_decay=0.9998, st_lr=0.0091, num_iters=300, **TEST,
        #                      content_hooks=["conv0.0"], style_hooks=["conv0.0"], lamb_style=3), name="s_dp")
        # plot_one(copy_update(idea, st_lr_decay=0.9998, st_lr=0.0091, num_iters=500, **TEST,
        #                      content_hooks=["conv2.0"], style_hooks=["conv1.0"], last_hook="conv2.0"), name="c_dp")
        # plot_one(copy_update(idea, st_lr_decay=0.9998, st_lr=0.0091, num_iters=400, style_loss="gram", **TEST), name="gram")
        # plot_one(copy_update(idea, st_lr_decay=0.9821, st_lr=0.8975, num_iters=500, lamb_content=0, **TEST), name="con")
        # plot_one(copy_update(idea_r, st_lr_decay=0.9821, st_lr=0.8975, num_iters=500, lamb_content=0, **TEST), name="sty")
        idea.update(
            count=20, metric="stval_perceptual", lr=False, st_lr=(5, 0.00001), st_lr_decay=(1.0, 0.7),
        )
        # sweep(**copy_update(idea, num_iters=5), name="i5")
        # sweep(**copy_update(idea, num_iters=10), name="i10")
        # sweep(**copy_update(idea, num_iters=25), name="i25")
        # sweep(**copy_update(idea, num_iters=100), name="i100")
        # sweep(**copy_update(idea, num_iters=500), name="i500")

    # MA-T
    if "gpm-train-mat" in tasks:
        mat = dict(
            **MA_T1, **LAMBS, epochs=50, lr_full=20, lr_decay=0.8845, lr=0.00036, lamb_kl=0.00001,
            layers=2, dim_feedforward=512, d_model=128, d_latent=8, d_latent_style=4
        )
        mat.update(
            count=10, metric="val_perceptual", lr=True, lr_decay=(1.0, 0.8), epochs=(10, 20, 50), lamb_kl=(10, 0.00001),
            layers=(2, 3, 4), dim_feedforward=(128, 256, 512), d_model=(32, 64, 128),
            d_latent=(1, 2, 4, 8), d_latent_style=(1, 2, 4, 8, 16)
        )
        # sweep(**mat)

        mat2 = dict(
            **MA_T2, **LAMBS, epochs=1, lr_delay=0, lr_full=4, lr_decay=1, lr=0.001, lamb_kl=0.0004,
            layers=2, dim_feedforward=512, d_model=128, d_latent=8, d_latent_style=4,
            # load_artifact="bert-tiny-bxk4p1j0:latest"
        )
        online(**mat2)
        # online(**mat2, skip_train=True, load_artifact="bert-tiny-3delxaf0:latest", num_iters=0, st_lr_decay=0.9981, st_lr=0.1354)
        # online(**mat2, skip_train=True, load_artifact="bert-tiny-i3o5kvs0:latest", num_iters=0, st_lr_decay=0.9981, st_lr=0.1354)
        methods_sm2sp["MA-T"] = copy_update(mat2, skip_train=True, load_artifact="bert-tiny-kearl7xi:latest")
        methods_sm2sp["MA-T-10"] = copy_update(mat2, skip_train=True, num_iters=10, st_lr_decay=0.9678, st_lr=0.1050,
                                               load_artifact="bert-tiny-kearl7xi:latest")
        methods_sm2sp["MA-T-25"] = copy_update(mat2, skip_train=True, num_iters=25, st_lr_decay=0.9514, st_lr=0.1143,
                                               load_artifact="bert-tiny-kearl7xi:latest")
        methods_sp2sm["MA-T"] = copy_update(mat2, skip_train=True, load_artifact="bert-tiny-kearl7xi:latest")
        methods_sp2sm["MA-T-10"] = copy_update(mat2, skip_train=True, num_iters=10, st_lr_decay=0.9042, st_lr=0.0651,
                                               load_artifact="bert-tiny-kearl7xi:latest")
        methods_sp2sm["MA-T-25"] = copy_update(mat2, skip_train=True, num_iters=25, st_lr_decay=0.8766, st_lr=0.0346,
                                               load_artifact="bert-tiny-kearl7xi:latest")
        # local(**copy_update(mat2, skip_train=True, load_artifact="bert-tiny-kearl7xi:latest", **TEST))
        # online(**copy_update(mat2, skip_train=True, num_iters=5, st_lr_decay=0.9981, st_lr=0.1354, load_artifact="bert-tiny-kearl7xi:latest", **TEST))
        # local(**copy_update(mat2, skip_train=True, num_iters=10, st_lr_decay=0.9678, st_lr=0.1050, load_artifact="bert-tiny-kearl7xi:latest", **TEST))
        # local(**copy_update(mat2, skip_train=True, num_iters=25, st_lr_decay=0.9514, st_lr=0.1143, load_artifact="bert-tiny-kearl7xi:latest", **TEST))
        # online(**copy_update(mat2, skip_train=True, num_iters=100, st_lr_decay=0.9705, st_lr=0.4917, load_artifact="bert-tiny-kearl7xi:latest", **TEST))
        # online(**copy_update(mat2, skip_train=True, num_iters=500, st_lr_decay=0.9919, st_lr=0.2625, load_artifact="bert-tiny-kearl7xi:latest", **TEST))
        # online(**copy_update(mat2, skip_train=True, num_iters=10, st_lr_decay=0.9042, st_lr=0.0651, load_artifact="bert-tiny-kearl7xi:latest", **TEST))  # r
        # online(**copy_update(mat2, skip_train=True, num_iters=25, st_lr_decay=0.8766, st_lr=0.0346, load_artifact="bert-tiny-kearl7xi:latest", **TEST))  # r
        # online(**copy_update(mat2, skip_train=True, num_iters=5, st_lr_decay=0.9061, st_lr=0.1298, load_artifact="bert-tiny-kearl7xi:latest", **TEST))
        # online(**copy_update(mat2, skip_train=True, num_iters=10, st_lr_decay=0.8555, st_lr=0.1310, load_artifact="bert-tiny-kearl7xi:latest", **TEST))
        # online(**copy_update(mat2, skip_train=True, num_iters=25, st_lr_decay=0.9851, st_lr=0.0696, load_artifact="bert-tiny-kearl7xi:latest", **TEST))
        # online(**copy_update(mat2, skip_train=True, num_iters=100, st_lr_decay=0.9652, st_lr=0.3142, load_artifact="bert-tiny-kearl7xi:latest", **TEST))
        # online(**copy_update(mat2, skip_train=True, num_iters=500, st_lr_decay=0.9948, st_lr=0.2115, load_artifact="bert-tiny-kearl7xi:latest", **TEST))

        # online(**copy_update(mat2, lamb_style=1, **TEST))
        # online(**copy_update(mat2, lamb_style=100, **TEST))

        # del mat2["load_artifact"]
        # online(**copy_update(mat2, epochs=10, **TEST))  # no_pt

        # online(**copy_update(mat2, skip_train=True, load_artifact="bert-tiny-5xkf35uc:latest", **TEST)) # ft2
        mat2_i = copy_update(mat2,
                             count=20, metric="stval_perceptual", skip_train=True, lr=False, st_lr_decay=(1.0, 0.8),
                             st_lr=(2, 0.001), st_lr_full=0, num_iters=10, load_artifact="bert-tiny-kearl7xi:latest",
                             name="t10"
                             )
        mat2.update(
            count=10, metric="stval_perceptual", lr=True, lr_decay=(1.0, 0.8), lamb_kl=(10, 0.00001),
        )
        # sweep(**mat2)
        # sweep(**copy_update(mat2_i, num_iters=5, name="t5"))
        # sweep(**copy_update(mat2_i, num_iters=10, name="t10"))
        # sweep(**copy_update(mat2_i, num_iters=25, name="t25"))
        # sweep(**copy_update(mat2_i, num_iters=100, name="t100"))
        # sweep(**copy_update(mat2_i, num_iters=500, name="t500"))

        # mat3 = dict(
        #     **MA_T3, **LAMBS, epochs=1, lr=0.00055, lamb_kl=0.0004, load_artifact="bert-tiny-kearl7xi:latest",
        #     layers=2, dim_feedforward=512, d_model=128, d_latent=8, d_latent_style=4
        # )
        # # local(**mat3)
        # mat3.update(
        #     count=10, metric="stval_perceptual", lr=True, lamb_kl=(0.1, 0.00001),
        # )
        # # sweep(**mat3)
