import itertools
import warnings

import torch
import numpy as np

from metrics import prd, predictive_utility, classify_style
from metrics import metric_dict as select_metric
from iterative import iterate
from tools import get_act
from models import InputOpt, TransformerAE, TransformerAE_Disentangled, BiLSTM_Disentangled, CNN_Disentangled
from data.speech import plot_sp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# evaluation of style transfer


def evaluate_style_transfer(data1, data2, model=None, fmodel=None, metrics=False, skip=None, one=False,
                            features=False, num_iters=0, name=None, **iter_kwargs):
    """
    evaluate style transfer on inference-time metrics like precision/recall, predictive utility

    :param data1: content data
    :param data2: style data
    :param model: model for model-based methods
    :param fmodel: feature model e.g. for data-based methods / iterative
    :param metrics: list of metrics ("PR": precision-recall, "MAE": predictive utility, "ACC": classification)
    :param features: if to use features instead of data (e.g. for precision-recall)
    :param num_iters: number of iterates for data-based / iterative methods
    :param iter_kwargs: parameters for iterations, includes [ch, sh, cl, sl, ls, lr]
    :param name: if not None execute test mode with tnse plots and returning prd
    """

    if not one:
        print("Evaluate on %d content samples and %d style samples" %
              (len(data1.sampler), len(data2.sampler)))
        n_samples = min(len(data1.sampler), len(data2.sampler))
    generated = []
    style = []
    content = []
    fgenerated = []
    fstyle = []
    fcontent = []
    iter_ = 0
    model.eval()
    for batch1, batch2 in zip(data1, itertools.cycle(data2)):
        if skip is not None and len(style) > skip:
            break

        # adapted size for last batch
        if len(batch2["x"]) < len(batch1["x"]):
            batch1 = {k: v[:len(batch2["x"])] for k, v in batch1.items()}
        if len(batch1["x"]) < len(batch2["x"]):
            batch2 = {k: v[:len(batch1["x"])] for k, v in batch2.items()}

        # style transfer
        if type(model) in [TransformerAE_Disentangled, BiLSTM_Disentangled, CNN_Disentangled]:
            latent2 = get_act(model, batch2, ["fc_mu", "fc_log_std", "fc_f_mu", "fc_f_log_std"], last_hook="fc_log_std")
            f_mu2 = latent2["fc_f_mu"]
            f_std2 = torch.exp(latent2["fc_f_log_std"])
            print("-----------------------------1------------IN AAA------------")
            with torch.no_grad():
                gen = model(batch1["x"].to(device), src_mask=torch.zeros_like(batch1["x"][:, :, 0]).to(device) == 1,
                            style=[f_mu2, f_std2])[0].cpu()
        elif type(model) in [TransformerAE]:
            print("-----------------------------2------------IN AAA------------")

            with torch.no_grad():
                gen = model(batch1["x"].to(device), src_mask=torch.zeros_like(batch1["x"][:, :, 0]).to(device) == 1).cpu()
        else:
            print("-----------------------------3------------IN AAA------------")

            gen = None


        # Improve with additional iterations
        print("---------------------------------")

        print(batch1["x"].shape)
        print("---------------------------------")
        if num_iters > 0:
            gen = iterate(fmodel, batch1["x"].to(device), batch2["x"].to(device), init=gen, num_iters=num_iters, **iter_kwargs)
            print("----------------Gen-----------------")

            print(gen)
            print(gen.shape)
            print("----------------Gen-----------------")
            if torch.isnan(gen).sum() > 1:
                warnings.warn("Nans detected")
                continue
        if one:
            if "mean" in batch1.keys() and data1.dataset.scale in ["scale", "scale-detrend"]:
                return gen - batch1["mean"]
            else:
                return gen

        # center non-centered data
        if "mean" in batch1.keys():
            if data1.dataset.scale in ["scale", "scale-detrend"]:
                fgenerated += [gen]
                generated += [gen - batch1["mean"]]
                fcontent += [batch1["x"]]
                content += [batch1["x"] - batch1["mean"]]
            else:
                generated += [gen]
                fgenerated += [gen + batch1["mean"]]
                content += [batch1["x"]]
                fcontent += [batch1["x"] + batch1["mean"]]
            if data2.dataset.scale in ["scale", "scale-detrend"]:
                fstyle += [batch2["x"]]
                style += [batch2["x"] - batch2["mean"]]
            else:
                style += [batch2["x"]]
                fstyle += [batch2["x"] + batch2["mean"]]
        else:
            generated += [gen]
            style += [batch2["x"]]
            content += [batch1["x"]]

        iter_ += 1
        print("[%d/%d]" % (iter_, max(len(data1), len(data2))))

    if name is not None:
        return prd(generated, style, name=name, t_SNE=True)[1]

    evals = {}
    for m in metrics:
        if m in ["PR", "MAE", "ACC"]:
            evals.update(select_metric[m](generated, style))
        else:
            metric_ = select_metric[m](fmodel, **(iter_kwargs.get("criterion_params", {})))
            metric_vals = {}
            if "mean" in batch1.keys() and m == "fin":
                gcs = zip(fgenerated, fcontent, fstyle)
            else:
                gcs = zip(generated, content, style)
            for g, c, s in gcs:
                for k, v in metric_(g.to(device), labels=c.to(device), style_labels=s.to(device)).items():
                    metric_vals[k] = metric_vals.get(k, []) + [len(g) * v]
            evals.update(**{k: sum(v) / n_samples for k, v in metric_vals.items()})
    return evals
