import logging
import os
from functools import partial

import numpy as np
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import torch
from torchvision.transforms import transforms

import wandb
from config import NETS_FOLDER
from data import data_dict as select_data
from data import transform_dict as select_transform
from models import model_dict as select_model
from visualize import Plotter_local, Plotter_wandb
from metrics import metric_dict
from style_transfer import evaluate_style_transfer

# global variables
log = logging.debug
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
plotter = Plotter_local()
save = print

PROJECT = "st_feature_one_channel"


# Evaluate a single metric on a dataset
def eval_(model, feature_model, data, metric, metric_params):

    if metric in metric_dict.keys():
        if "perceptual" in metric:
            metric_ = metric_dict[metric](feature_model, **metric_params)
        else:
            metric_ = metric_dict[metric](**metric_params)
    else:
        raise NotImplementedError("Metric")

    metric_vals = []
    model.eval()
    with torch.no_grad():
        for batch in data:
            inputs = batch["x"].to(device)
            labels = batch["y"].to(device)
            mask = (batch["mask"] == 1).to(device)
            outputs = model(inputs, src_mask=mask)
            metric_vals += [metric_(outputs, labels)[metric].item()]
    model.train()

    return np.mean(metric_vals)


# Evaluate a set of metrics on a dataset
def all_eval_(model, feature_model, data, metrics, metric_params, **model_kwargs):

    metrics_ = []
    for metric in metrics:
        if metric in metric_dict.keys():
            if "perceptual" in metric:
                metrics_ += [metric_dict[metric](feature_model, **metric_params)]
            else:
                metrics_ += [metric_dict[metric](**metric_params)]
        else:
            raise NotImplementedError("Metric")

    metric_vals = {}
    model.eval()
    with torch.no_grad():
        for batch in data:
            inputs = batch["x"].to(device)
            labels = batch["y"].to(device)
            if "mask" in batch.keys():
                mask = (batch["mask"] == 1).to(device)
            else:
                mask = (torch.zeros_like(inputs[:, :, 0]) == 1).to(device)
            if "style_perm" in model_kwargs.keys() and inputs.shape[0] != model_kwargs["style_perm"].shape[0]:
                print("skipped batch")
                continue
            outputs = model(inputs, src_mask=mask, **model_kwargs)
            ths_vals = {}
            for metric_ in metrics_:
                try:
                    ths_vals.update(metric_(outputs, labels, cluster_idxs=batch["didx"]))
                except (KeyError, TypeError) as e:
                    ths_vals.update(metric_(outputs, labels))
            for k in ths_vals.keys():
                metric_vals[k] = metric_vals.get(k, []) + [ths_vals[k].item()]
    model.train()

    avg = {}
    for k, vals in metric_vals.items():
        if "_" not in k or "len" in k:
            avg[k] = np.mean(vals)
        else:
            # weighted average w.r.t. occurrence
            c = int(k.split("_")[-1])
            lens = metric_vals["len_%d" % c]
            avg[k] = np.sum((np.array(vals) * np.array(lens))) / np.sum(lens)
    return avg


def train_(model, train, criterion, optimizer, epochs, optim_params=None, criterion_params=None, feature_model=None,
           val=None, metrics=None, save_path=None, final_save_path=None, shuffle_style=False):
    """
    Train a model on training data

    :param model: PyTorch model
    :param feature_model: pretrained PyTorch model to obtain features for perceptual loss
    :param train: PyTorch dataloader
    :param val: PyTorch dataloader, optional
    :param criterion: str, loss
    :param optimizer: str
    :param epochs: int, number of epochs
    :param optim_params: dict, additional parameters to the optimizer, e.g. learning rate, optional
    :param criterion_params: dict, additional parameters to the loss, e.g. weights, optional
    :param metrics: list of str, additional metrics, e.g. validation loss, optional
    :param save_path: str, where to save checkpoints, optional
    :param save_path: str, where to save final model, optional
    :param shuffle_style: boolean, if to permutate the style latent/loss
    """

    if criterion in metric_dict.keys():
        if "perceptual" in criterion:
            loss_func = metric_dict[criterion](feature_model, **criterion_params)
        else:
            loss_func = metric_dict[criterion](**criterion_params)
    elif criterion == "cloze-MSE":
        loss_func = torch.nn.MSELoss()
    else:
        raise NotImplementedError("Loss")

    # MSE at clozed positions only (prediction > reconstruction)
    # todo: why not mse loss everywhere with weights?
    clozed_only = criterion in ["cloze-MSE"]

    if optim_params is None:
        optim_params = dict()
    if optimizer == "Adam":
        lr = optim_params.get("lr", 0.001)
        lr_decay = optim_params.get("lr_decay", 1)
        lr_delay = optim_params.get("lr_delay", 0)
        lr_warmup = optim_params.get("lr_warmup", 0)
        lr_full = optim_params.get("lr_full", 0)
        beta1 = optim_params.get("beta1", 0.9)
        beta2 = optim_params.get("beta2", 0.999)
        weight_decay = optim_params.get("weight_decay", 0)
        optimizer = torch.optim.Adam(model.parameters(), betas=(beta1, beta2), lr=lr, weight_decay=weight_decay)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
        def rate(t):
            if t < lr_delay:
                return 0
            elif lr_delay <= t < lr_delay + lr_warmup:
                if t == lr_delay:
                    print("Warmup starting")
                return t/lr_warmup
            elif lr_delay + lr_warmup <= t < lr_delay + lr_warmup + lr_full:
                if t == lr_delay + lr_warmup:
                    print("Warmup done")
                return 1
            else:
                if t == lr_delay + lr_warmup + lr_full and lr_decay < 1:
                    print("Cooling starting")
                return lr_decay ** (t - lr_warmup - lr_full)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=rate)
    else:
        raise NotImplementedError("Optimizer")

    # Training loop
    iters = 0
    loss_sum = 0
    print("Starting Training Loop...")
    for epoch in range(1, int(np.ceil(epochs)) + 1):
        for i, batch in enumerate(train, 1):
            inputs = batch["x"].to(device)
            labels = batch["y"].to(device)
            if "mask" in batch.keys():
                mask_keys = batch["mask"]
            else:
                mask_keys = torch.zeros_like(inputs[:, :, 0])
            mask = (mask_keys == 1).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if shuffle_style is not None:
                if shuffle_style == "full":
                    perm = torch.randperm(inputs.shape[0])
                elif shuffle_style == "half":
                    b = inputs.shape[0]
                    subperm = torch.randperm(b)[:(b // 2)]
                    perm = torch.arange(b)
                    perm[torch.sort(subperm)[0]] = subperm
                elif shuffle_style == "c1":
                    b = inputs.shape[0]
                    cidx = batch["didx"] == 1
                    if sum(cidx) > 0:
                        perm = torch.cat([torch.where(cidx)[0][torch.randperm(sum(cidx))]
                                          for _ in range(int(torch.ceil(b / sum(cidx))))], axis=0)[:b]
                    else:
                        subperm = torch.randperm(b)[:(b // 2)]
                        perm = torch.arange(b)
                        perm[torch.sort(subperm)[0]] = subperm
                else:
                    raise ValueError("Unknown shuffle: %s" % shuffle_style)
                outputs = model(inputs, src_mask=mask, style_perm=perm)
            else:
                outputs = model(inputs, src_mask=mask)

            if clozed_only:
                cloze = (mask_keys > 0).to(device)
                outputs = outputs[cloze]
                labels = labels[cloze]

            # from inference import analyze, inference
            # m = min(torch.sum(batch["didx"] == 1), torch.sum(batch["didx"] == 0))
            # b1 = {k: v[batch["didx"] == 1][:m] for k, v in batch.items()}
            # b2 = {k: v[batch["didx"] == 0][:m] for k, v in batch.items()}
            # inference(None, data=dict(model=model, batch1=b1, batch2=b2, criterion=loss_func.vae_perceptual.perceptual))
            # analyze(None, data=dict(model=model, batch=batch, criterion=loss_func.vae_perceptual.perceptual), perm=perm)

            try:
                # loss \w cluster information (e.g. disentangled loss) or \w permutation
                if shuffle_style:
                    train_metrics = loss_func(outputs, labels, cluster_idxs=batch["didx"], style_perm=perm)
                else:
                    train_metrics = loss_func(outputs, labels, cluster_idxs=batch["didx"])
            except (KeyError, TypeError) as e:
                train_metrics = loss_func(outputs, labels)

            loss = train_metrics.pop(criterion)
            loss.backward()
            optimizer.step()

            # Output training stats
            loss = loss.item()
            loss_sum += loss
            train_perc = ((epoch - 1) * len(train) + i) / (epochs * len(train))
            log({'epoch_': train_perc * epochs, 'loss': loss})
            iters += 1
            for k, v in train_metrics.items():
                log({'epoch_': train_perc * epochs, 'train_' + k: v})
            if iters % 10 == 0:
                print('[%.2f %%][%d/%d][%d/%d]\tLoss: %.4f' %
                      (100 * train_perc, epoch, epochs, i, len(train), loss_sum / 10))
                loss_sum = 0

            # allow fractional epochs
            if epoch - 1 + i / len(train) > epochs:
                break
        scheduler.step()

        if save_path is not None:
            if epoch % 25 == 0 and epoch < epochs:
                # do checkpointing
                save(model.state_dict(), '%s/epoch_%d.cp' % (save_path, epoch))

        # if metrics is not None:
        #     for metric in metrics:
        #         dataset, method = metric.split("_")
        #         eval = eval_(model, train if dataset == train else val, method, metric_params=criterion_params)
        #         print("%s: %.4f" % (metric, eval))
        #         log({"epoch": epoch, metric: eval})

        # Split metrics by dataset and evaluate at once
        if metrics is not None:
            metric_by_set = {}
            for metric in metrics:
                dataset, method = metric.split("_")
                metric_by_set[dataset] = metric_by_set.get(dataset, []) + [method]

            for dataset, methods in metric_by_set.items():
                evals = all_eval_(model, feature_model, train if dataset == train else val, methods, metric_params=criterion_params)
                for method, eval in evals.items():
                    metric = dataset + "_" + method
                    print("%s: %.4f" % (metric, eval))
                    log({"epoch": epoch, metric: eval})

    # final checkpoint
    if final_save_path is not None:
        save(model.state_dict(), '%s/epoch_%d.cp' % (final_save_path, epoch))


def plot_predict(model, data, title="Reconstruction", id="reconstruct", k=1, transform=None):
    """
    Plot one predicted vs actual time series
    :param model: PyTorch model
    :param data: PyTorch dataloader
    :param title: title of plot
    :param id: plot id, only needed for wandb
    :param k: number of samples (for VAE)
    :return:
    """
    # first sample as numpy
    sample = data.dataset[data.sampler.indices[0]]
    masked = sample["x"]
    truth = sample["y"]
    masked_torch = torch.from_numpy(masked[None]).to(device)
    truth_torch = torch.from_numpy(truth[None]).to(device)
    if "mask" in sample.keys():
        mask = sample["mask"] > 0
        mask_torch = torch.from_numpy(mask[None]).to(device)
    else:
        mask_torch = torch.full(masked_torch[:, :, 0].shape, False, device=device)

    preds = []
    model.eval()
    for _ in range(k):
        with torch.no_grad():
            pred = model(masked_torch, src_mask=mask_torch)
            # vae compatibility
            if type(pred) == tuple:
                pred = pred[0]
            mse = torch.nn.MSELoss()(pred, truth_torch).item()
            preds += [pred]
            print("MSE of sample: %.3g" % mse)
    model.train()

    x = np.arange(masked.shape[0])
    multi = masked.shape[1] > 1 and type(plotter) == Plotter_local

    if multi:
        plotter.combine([id + str(channel) for channel in range(masked.shape[1])])
    for channel in range(min(masked.shape[1], 2)):
        plotter.plot_lines(x, [masked[:, channel], truth[:, channel]] +
                           [np.array(pred[0, :, channel].cpu()) for pred in preds],
                           silent=multi and channel < masked.shape[1],
                           labels=["masked", "ground truth", "reconstruction"], title=title, id=id + str(channel))
    if transform is not None:
        if multi:
            plotter.combine([id + str(channel) + "tf" for channel in range(masked.shape[1])])
        for channel in range(min(masked.shape[1], 2)):
            plotter.plot_lines(x, [transform(masked[:, channel]), transform(truth[:, channel])] +
                               [transform(np.array(pred[0, :, channel].cpu())) for pred in preds],
                               silent=multi and channel < masked.shape[1],
                               labels=["masked", "ground truth", "reconstruction"], title=title, id=id + str(channel) + "tf")


def load_artifact(artifact):
    """ Load W&B artifact by name """
    wandb.init(project=PROJECT, job_type="explore")
    model_artifact = wandb.run.use_artifact(artifact)

    dir = NETS_FOLDER + "temp"
    # delete old temp files
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    # download and find checkpoint
    model_artifact.download(root=dir)
    filenames = next(os.walk(dir), (None, None, []))[2]
    newest = sorted(filenames, reverse=True, key=lambda x: os.path.getmtime(os.path.join(dir, x)))[0]
    return os.path.join(dir, newest)


def load_data(dataset, params):
    """ load data from config """
    if dataset is None:
        return None, None, None
    if "transform" in params.keys():
        # Masking
        tf = select_transform[params["transform"]](**params["transform_params"])
        params["transform"] = transforms.Compose([transforms.Lambda(tf), ])
    train, val, test = select_data[dataset](**params).dataloader()

    return train, val, test


def load_net_and_data(config):
    """ Load model(s) and data from config """

    # Data
    train, val, test = load_data(config["dataset"], config["dataset_params"])
    # Load style data to metric
    if "style_dataset" in config.keys():
        # print("Loading style data")
        # print("---------------------------------------------------")
        # print(config["style_dataset"])
        # print("---------------------------------------------------")

        # print(config["style_dataset_params"])
        # print("---------------------------------------------------")

        style_data, _, _ = load_data(config["style_dataset"], config["style_dataset_params"])
        config["training_params"]["criterion_params"]["style_dataset"] = style_data

    # Net
    model = select_model[config["model"]](**config["model_params"]).to(device)
    print("%d trainable parameters" % sum([p.numel() for p in model.parameters() if p.requires_grad]))
    feature_model = None
    if "feature_model" in config.keys():
        feature_model = select_model[config["feature_model"]](**config["feature_model_params"]).to(device)

    # Net parameters
    if "load_artifact" in config.keys() or "load_path" in config.keys():
        path = load_artifact(config["load_artifact"]) if "load_artifact" in config.keys() else config["load_path"]
        print("Loading from: " + path)
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            print("parameters loaded")
        except FileNotFoundError:
            print("No parameters loaded")
            pass
    if "feature_model" in config.keys():
        path = load_artifact(config["feature_load_artifact"]) if "feature_load_artifact" in config.keys() else config["feature_load_path"]
        print("Loading features from: " + path)
        try:
            load_state = torch.load(path, map_location=device)
            load_state = {k: v for k, v in load_state.items() if k in feature_model.state_dict().keys()}
            feature_model.load_state_dict(load_state)
            print("parameters loaded on feature net")
        except FileNotFoundError:
            # Features have to be loaded
            raise FileNotFoundError("No parameters loaded on feature net")

    return model, feature_model, train, val, test


def train_model(config):
    """
    Everything before and after training:
    Loading, settings, validating style transfer, testing
    """

    # Model and data
    model, feature_model, train, val, test = load_net_and_data(config)

    # Checkpoints
    global save
    if "save_path" in config.keys():
        if not os.path.exists(config["save_path"]):
            os.mkdir(config["save_path"])
        if config.get("wandb", False):
            basename = os.path.basename(config["save_path"]) + "-" + wandb.run._run_id

            def make_artifact(name=basename):
                return wandb.Artifact(name, type="model")

            def save(cp, file, art=make_artifact):
                torch.save(cp, file)
                artifact = art()
                artifact.add_file(file)
                wandb.run.log_artifact(artifact)
        else:

            def save(cp, file):
                torch.save(cp, file)

    # Logging
    global log
    global plotter
    if config.get("wandb", False):
        log = wandb.log
        plotter = Plotter_wandb()
    else:
        log = lambda x: logging.debug("    ".join(["%s: %g" % (k, v) for k, v in x.items()]))
        plotter = Plotter_local()

    # checkpoints / metrics
    metrics = config.get("metrics", None)
    metrics_val = [m for m in metrics if m.startswith("val") or m.startswith("train")]
    metrics_stval = [m for m in metrics if m.startswith("stval")]
    metrics_sttest = [m for m in metrics if m.startswith("sttest")]
    metrics_test = [m for m in metrics if m.startswith("test")]
    final_save_path = config.get("save_path", None)
    save_path = final_save_path if not config.get("wandb", True) else None

    if not config.get("skip_train", False):
        train_(model, feature_model=feature_model, train=train, val=val, **config["training_params"],
               metrics=metrics_val, save_path=save_path, final_save_path=final_save_path)

        # visualize
        plot_predict(model, train, title="Reconstruction (train)", id="reconstruct_train")
        plot_predict(model, val, title="Reconstruction (val)", id="reconstruct_val")

    # Load style transfer data and val/test
    if "stc_dataset" in config.keys() and "sts_dataset" in config.keys():
        print("Loading style transfer data")
        _, content_val, content_test = load_data(config["stc_dataset"], config["stc_dataset_params"])
        _, style_val, style_test = load_data(config["sts_dataset"], config["sts_dataset_params"])
        if len(metrics_stval) > 0:
            evals = evaluate_style_transfer(
                content_val, style_val, model=model, fmodel=feature_model,
                metrics=[m.split("_")[1] for m in metrics_stval], **config["st_iter_params"]
            )
            for method, eval in evals.items():
                metric = "stval_" + method
                print("%s: %.4g" % (metric, eval))
                log({metric: eval})
        if len(metrics_sttest) > 0:
            evals = evaluate_style_transfer(
                content_test, style_test, model=model, fmodel=feature_model,
                metrics=[m.split("_")[1] for m in metrics_sttest], **config["st_iter_params"]
            )
            for method, eval in evals.items():
                metric = "sttest_" + method
                print("%s: %.4g" % (metric, eval))
                log({metric: eval})


    # test
    metrics_test = [m.split("_")[1] for m in metrics_test]
    if len(metrics_test) > 0:
        evals = all_eval_(model, feature_model, test, metrics_test,
                          metric_params=config["training_params"]["criterion_params"])
        for method, eval in evals.items():
            metric = "test_" + method
            print("%s: %.4g" % (metric, eval))
            log({metric: eval})
