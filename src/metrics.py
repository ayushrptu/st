import warnings
from functools import partial

import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import torch
import matplotlib.pyplot as plt

from tools import vae_compatible_call, StopForward, Freeze, rescale, InfIter
from data import gp_likelihood
# from fast_soft_sort import soft_rank
from extra import compute_prd_from_embedding
# from plot import dimred

# Metrics and Losses
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# mse
def mse(x, y, keepdims=0):
    return torch.mean((x - y) ** 2, tuple(range(keepdims, len(x.shape))))


# mse on gram matrices of features
def mse_gram(x, y, keepdims=0):
    if len(x.shape) > 3:
        x = x.reshape(x.shape[0], x.shape[1], -1)
        y = y.reshape(y.shape[0], y.shape[1], -1)
    # todo: guarantee that x is in shape [batch, channels, seq_len] (or similarly fixed)
    g1 = torch.matmul(x, x.transpose(1, 2)) / x.shape[2]
    g2 = torch.matmul(y, y.transpose(1, 2)) / y.shape[2]
    return mse(g1, g2, keepdims)


# mse on mean, std of features
def mse_mean_std(x, y, keepdims=0):
    # todo: guarantee that x is in shape [batch, channels, seq_len] (or similarly fixed)
    m1 = torch.mean(x, dim=2)
    m2 = torch.mean(y, dim=2)
    s1 = torch.std(x, dim=2)
    s2 = torch.std(y, dim=2)
    return mse(m1, m2, keepdims) + mse(s1, s2, keepdims)


class Metric:
    def __init__(self, **kwargs):
        """
        :param kwargs: metric parameters like weights
        """
        pass

    def __init__(self, outputs, labels):
        raise NotImplementedError


class MSE_Loss(Metric):
    def __init__(self, **kwargs):
        pass

    @vae_compatible_call
    def __call__(self, outputs, labels):
        return {"MSE": mse(outputs, labels)}


class KL_Loss(Metric):
    # todo: use hooks instead of hard-coding into model
    def __init__(self, **kwargs):
        pass

    def __call__(self, outputs, labels):
        outputs, kl = outputs
        return {"KL": torch.mean(kl)}


class VAE_MSE_Loss(Metric):
    # todo: Kloft ML II: More samples for reconstruction loss? E[MSE] after all!
    def __init__(self, lamb_recon=1, lamb_kl=1, **kwargs):
        self.l = rescale([lamb_recon, lamb_kl])

    def __call__(self, outputs, labels):
        outputs, kl = outputs
        mse_ = mse(outputs, labels)
        kl_ = torch.mean(kl)
        return {"VAE_MSE": self.l[0] * mse_ + self.l[1] * kl_,
                "MSE": mse_, "KL": kl_}


# Content and Style loss based on moving average and log-return statistics
class Finance_Loss:
    def __init__(self, *args, lamb_content=1, lamb_style=1, N=15, **kwargs):
        """
        :param N: window size of sliding window as percentage of sequence length
        """
        self.N = N
        self.l = rescale([lamb_content, lamb_style])

    def trend(self, outputs, labels):
        """ moving average MSE """

        def roll_mean(x):
            return sum([x[:, (self.N - 1 + shift):(x.shape[1] - self.N + 1 + shift), :]
                        for shift in range(self.N)]) / self.N

        out_trend = roll_mean(outputs)
        labels_trend = roll_mean(labels)
        return mse(out_trend, labels_trend)

    def ac(self, x1, x2):
        """ MSE of autocorrelation """

        # def slow_acf(x):
        #     mean = torch.mean(x, 1, keepdim=True)
        #     sumvar = torch.sum((x - mean) ** 2, 1)
        #     x = x - mean
        #     return torch.stack([torch.ones(x.shape[0], x.shape[2]).to(x.dtype).to(device)] + [torch.sum(x[:, :-i, ] * x[:, i:, :], 1) / sumvar for i in range(1, x.shape[1])], 1)

        def acf(x):
            mean = torch.mean(x, 1, keepdim=True)
            sumvar = torch.sum((x - mean) ** 2, 1)
            x = x - mean
            x_ = x.swapaxes(1, 2)
            x_ = torch.matmul(x_.unsqueeze(3), x_.unsqueeze(2))
            return torch.stack([torch.ones(x.shape[0], x.shape[2]).to(x.dtype).to(device)] + [torch.sum(torch.diagonal(x_, i, 2, 3), 2) / sumvar for i in range(1, x.shape[1])], 1)

        a1 = acf(x1)
        a2 = acf(x2)
        return mse(a1, a2), a1, a2

    def vol(self, x1, x2):
        """ volatility """
        v1 = torch.std(x1, 1)
        v2 = torch.std(x2, 1)
        return mse(v1, v2)

    @staticmethod
    def fft(a1, a2):
        """ compute the error in the Fourier domain, for auto-correlations this is the power spectral density """
        p1 = torch.fft.fft(a1, dim=1)
        p2 = torch.fft.fft(a2, dim=1)
        return torch.mean((p1 - p2).abs()**2)

    @staticmethod
    def slogret(x):
        """ signed log return with slog(0) = 0 """
        x = x[:, 1:] / x[:, :-1]
        if torch.any(x) <= 0:
            warnings.warn("Negative returns detected, using signed log")
            raise ValueError("Negative returns")
        out = torch.zeros_like(x)
        out[x != 0] = torch.log(torch.abs(x[x != 0])) * torch.sign(x)[x != 0]
        return out

    @vae_compatible_call
    def __call__(self, outputs, labels, style_labels=None, cluster_idxs=None):
        """
        Compute a style loss between outputs and labels

        :param cluster_idxs: if given report loss by cluster id and combined
        """
        # todo: clsuter_idxs
        if style_labels is None:
            style_labels = labels
        loss_trend = self.trend(outputs, labels)
        lret_out = self.slogret(outputs)
        lret_lab = self.slogret(style_labels)
        loss_ac, ac_out, ac_lab = self.ac(lret_out, lret_lab)
        loss_vol = self.vol(lret_out, lret_lab)
        loss_fft = self.fft(ac_out, ac_lab)
        loss_style = (loss_ac + loss_vol + loss_fft) / 3

        res = {}
        res.update({"fin": self.l[0] * loss_trend + self.l[1] * loss_style,
                    "trendfin": loss_trend, "stylefin": loss_style,
                    "acfin": loss_ac, "volfin": loss_vol, "fftfin": loss_fft})
        return res


class Cov_Loss:
    def __init__(self, *args, lamb_content=1, lamb_style=1, N=15, **kwargs):
        self.x = None
        self.Sig1 = None
        self.Sig2 = None

    @vae_compatible_call
    def __call__(self, outputs, labels=None, style_labels=None, cluster_idxs=None):
        """
        Compute a style loss between outputs and labels

        :param cluster_idxs: if given report loss by cluster id and combined
        """
        if self.x is None:
            self.x = np.linspace(0, 0.05 * outputs.shape[1], outputs.shape[1]).reshape(-1, 1)
            self.Sig1 = torch.tensor((RBF() + WhiteKernel(0))(self.x)).to(device)
            self.Sig2 = torch.tensor((RBF() + WhiteKernel(0.5))(self.x)).to(device)

        cov_data = torch.cov(outputs.squeeze(2).T)
        cov_sm = mse(cov_data, self.Sig1)
        cov_sp = mse(cov_data, self.Sig2)
        res = {"covsm": cov_sm, "covsp": cov_sp}
        return res


# Content and Style loss based on co-occurrence (gram matrices)
class Perceptual_Loss:
    # Shared collection over all instances
    feats = {}

    def __init__(self, model, content_hooks, style_hooks=None, last_hook=None,
                 lamb_content=1, lamb_style=1, style_loss="gram",
                 style_dataset=None, **kwargs):
        """
        :param content_hooks: list of str - where to compute content loss
        :param style_hooks: list of str - where to compute style loss, defaults to content_hooks
        :param last_hook: str - last hook, after which computation is stopped, optional
        :param style_loss: which style loss to use:
            "gram": align gram matrices (feature correlation)
            "mean_std", align mean, std (feature representation)
        :param style_dataset: PyTorch dataloader - align style with external dataset, not input
        """
        self.model = model
        assert style_loss in ["gram", "mean_std"]
        # todo: sqrt?

        # Save features after these models and stop forward pass after the module in last_hook
        self.content_hooks = content_hooks
        self.style_hooks = style_hooks or content_hooks
        self.hooks = list(set(self.content_hooks) | set(self.style_hooks))
        self.last_hook = last_hook

        # Parameters for style loss
        self.style_loss = style_loss
        self.l = rescale([lamb_content, lamb_style])

        self.style_dataset = InfIter(style_dataset) if style_dataset is not None else None

        # Attach hooks for feature activations
        def save_hook(m, inp, op, feats, m_name, stop=False):
            # Only collect in evaluation, i.e. when requires_grad is set to False for all parameters of the layer
            if np.all([p.requires_grad is False for p in m.parameters()]):
                feats[m_name] = op
                if stop:
                    raise StopForward

        for m in self.hooks:
            dict(self.model.named_modules())[m].register_forward_hook(
                partial(save_hook, feats=self.feats, m_name=m, stop=m == self.last_hook)
            )

    def _get_activations(self, outputs=None, labels=None, style_labels=None):
        """
        collect activations in self.in_feats/out_feats/style_feats

        style_perm: permutate labels for style features
        """
        # Freeze to activate collections hooks
        # Style features
        if style_labels is not None or self.style_dataset is not None:
            if style_labels is None:
                style_labels = next(self.style_dataset)["y"]

            # Align batch sizes
            if labels.shape[0] < style_labels.shape[0]:
                style_labels = style_labels[:labels.shape[0]]
            elif labels.shape[0] > style_labels.shape[0]:
                reps = int(np.ceil(labels.shape[0] / style_labels.shape[0]))
                style_labels = style_labels.repeat(reps, 1, 1)[:labels.shape[0]]

            with Freeze(self.model):
                # No gradients needed on label activations
                with torch.no_grad():
                    try:
                        mask = torch.zeros((style_labels.shape[0], style_labels.shape[1])).to(style_labels.dtype).to(device)
                        _ = self.model(style_labels.to(device), src_mask=mask)
                    except StopForward:
                        pass
                    self.style_feats = self.feats.copy()
        else:
            self.style_feats = None
        if labels is not None:
            # Input features
            with Freeze(self.model):
                # No gradients needed on label activations
                with torch.no_grad():
                    try:
                        mask = torch.zeros((labels.shape[0], labels.shape[1])).to(labels.dtype).to(device)
                        _ = self.model(labels.to(device), src_mask=mask)
                    except StopForward:
                        pass
                    self.in_feats = self.feats.copy()
        # Output features
        if outputs is not None:
            with Freeze(self.model):
                try:
                    mask = torch.zeros((outputs.shape[0], outputs.shape[1])).to(outputs.dtype).to(device)
                    _ = self.model(outputs.to(device), src_mask=mask)
                except StopForward:
                    pass
                self.out_feats = self.feats.copy()
            self.feats.clear()

    def content_style_loss(self, in_feats, out_feats, style_feats, style_labels, style_perm=None, mask=None):
        loss_content = 0
        loss_style = 0
        for m in self.content_hooks:
            in_ft = in_feats[m]
            out_ft = out_feats[m]
            if mask is not None:
                in_ft = in_ft[mask]
                out_ft = out_ft[mask]
            loss_content += mse(in_ft, out_ft)
        for m in self.style_hooks:
            ft = style_feats[m] if (style_labels is not None or self.style_dataset is not None) else in_feats[m]
            out_ft = out_feats[m]
            if style_perm is not None:
                ft = ft[style_perm]
            if mask is not None:
                ft = ft[mask]
                out_ft = out_ft[mask]

            if self.style_loss == "gram":
                loss_style += mse_gram(ft, out_ft)
            else:
                loss_style += mse_mean_std(ft, out_ft)
        loss_content /= len(self.content_hooks)
        loss_style /= len(self.style_hooks)

        return loss_content, loss_style

    @vae_compatible_call
    def __call__(self, outputs, labels, style_labels=None, cluster_idxs=None, style_perm=None):
        """
        Compute a style loss between outputs and labels

        :param cluster_idxs: if given report loss by cluster id and combined
        """
        # Set self.in_feats / self.out_feats / self.style_feats
        self._get_activations(outputs, labels, style_labels)

        res = {}
        if cluster_idxs is not None:
            ths_idx = np.unique(cluster_idxs)
            loss_content, loss_style = 0, 0
            for c in ths_idx:

                # select cluster samples
                mask = cluster_idxs == c

                ths_content, ths_style = self.content_style_loss(
                    self.in_feats, self.out_feats, self.style_feats, style_labels, style_perm, mask
                )
                res.update({"perceptual_%d" % c: self.l[0] * ths_content + self.l[1] * ths_style,
                            "content_%d" % c: ths_content, "style_%d" % c: ths_style, "len_%d" % c: sum(mask)})
                loss_content += sum(mask) * ths_content
                loss_style += sum(mask) * ths_style
            loss_content /= outputs.shape[0]
            loss_style /= outputs.shape[0]
        else:
            loss_content, loss_style = self.content_style_loss(self.in_feats, self.out_feats, self.style_feats,
                                                               style_labels, style_perm=style_perm)

        res.update({"perceptual": self.l[0] * loss_content + self.l[1] * loss_style,
                    "content": loss_content, "style": loss_style})
        return res

    def intra_content_style_loss(self, style_perm=None, loc="out"):
        """
        intra-batch content/style loss
        can be called after _get_activations - useful for DIS-perceptual

        loc: "out" - loss at output
             "in"  - loss on label (input) [opt. w/ perm]
        """
        intra_content = 0
        intra_style = 0
        for m in self.content_hooks:
            ft = self.out_feats[m] if loc == "out" else self.in_feats[m]
            b = ft.shape[0]
            outh = ft.unsqueeze(0).expand(b, -1, -1, -1)
            outv = ft.unsqueeze(1).expand(-1, b, -1, -1)
            intra_content += mse(outh, outv, 2)
        for m in self.style_hooks:
            ft = self.out_feats[m] if loc == "out" else self.in_feats[m]
            if style_perm is not None:
                if loc == "out":
                    pass  # ft = ft[torch.argsort(style_perm)]
                else:
                    ft = ft[style_perm]
            b, l, c = ft.shape
            outh = ft.unsqueeze(0).expand(b, -1, -1, -1).reshape(b ** 2, l, c)
            outv = ft.unsqueeze(1).expand(-1, b, -1, -1).reshape(b ** 2, l, c)
            if self.style_loss == "gram":
                intra_style += mse_gram(outh, outv, 1).reshape(b, b)
            else:
                intra_style += mse_mean_std(outh, outv, 1).reshape(b, b)
        intra_content /= len(self.content_hooks)
        intra_style /= len(self.style_hooks)

        return intra_content, intra_style


class VAE_Perceptual_Loss(Metric):
    def __init__(self, model, lamb_content=1, lamb_style=1, lamb_kl=1, **kwargs):
        self.l = rescale([lamb_content, lamb_style, lamb_kl])
        self.perceptual = Perceptual_Loss(model, lamb_content=lamb_content, lamb_style=lamb_style, **kwargs)

    def __call__(self, outputs, labels, cluster_idxs=None, style_perm=None):
        outputs, sv = outputs
        kl = sv["kl"]
        results = self.perceptual(outputs, labels, cluster_idxs=cluster_idxs, style_perm=style_perm)
        content_ = results["content"]
        style_ = results["style"]
        kl_ = torch.mean(kl)
        results.update({"VAE-perceptual": self.l[0] * content_ + self.l[1] * style_ + self.l[2] * kl_, "KL": kl_})
        if cluster_idxs is not None:
            for c in np.unique(cluster_idxs):
                content_ = results["content_%d" % c]
                style_ = results["style_%d" % c]
                kl_ = torch.mean(kl[cluster_idxs == c])
                results.update(
                    {"VAE-perceptual_%d" % c: self.l[0] * content_ + self.l[1] * style_ + self.l[2] * kl_,
                     "KL_%d" % c: kl_})
        return results


# Compute inter-batch latent distances
def intra_latent_dist(mu1, std1, mu2, std2):
    """
    compute latent distance as elementwise E[MSE] based on

    X_1 ~ N(mu1, std1**2)
    X_2 ~ N(mu2, std2**2), id of X1
    Then E[(X_1 - X_2)**2] = Var(X_1 - X_2) + E[X_1 - X_2]**2 = std1**2 + std2**2 + (mu1 - mu2)**2
    """
    b, ml, mc = mu1.shape
    _, sl, sc = mu2.shape

    # global/style
    mu1h = mu1.unsqueeze(0).expand(b, -1, -1, -1)
    mu1v = mu1.unsqueeze(1).expand(-1, b, -1, -1)
    std1h = std1.unsqueeze(0).expand(b, -1, -1, -1)
    std1v = std1.unsqueeze(1).expand(-1, b, -1, -1)

    # dynamics/content
    mu2h = mu2.unsqueeze(0).expand(b, -1, -1, -1)
    mu2v = mu2.unsqueeze(1).expand(-1, b, -1, -1)
    std2h = std2.unsqueeze(0).expand(b, -1, -1, -1)
    std2v = std2.unsqueeze(1).expand(-1, b, -1, -1)

    intra_f = torch.mean(std1h ** 2 + std1v ** 2 + (mu1h - mu1v) ** 2, tuple(range(2, len(mu1.shape) + 1)))
    intra_z = torch.mean(std2h ** 2 + std2v ** 2 + (mu2h - mu2v) ** 2, tuple(range(2, len(mu1.shape) + 1)))

    return intra_f, intra_z


class VAE_Perceptual_Disentangled_Loss(Metric):
    """
    VAE Perceptual Loss which enforces loss guided latent space structure
    through a strong correlation between latent distance and loss
    """
    def __init__(self, model, lamb_content=1, lamb_style=1, lamb_dis=1, lamb_kl=1, **kwargs):
        self.l = rescale([lamb_content, lamb_style, lamb_dis, lamb_kl])
        self.vae_perceptual = VAE_Perceptual_Loss(
            model, lamb_content=lamb_content, lamb_style=lamb_style, lamb_kl=lamb_kl, **kwargs
        )

    def __call__(self, outputs, labels, cluster_idxs=None, style_perm=None):
        """
        cluster_idxs: cluster id, when training on more than one dataset
        style_perm: permute the labels for style loss, e.g. when the same is done in the style latent space
        """
        results = self.vae_perceptual(outputs, labels, cluster_idxs=cluster_idxs, style_perm=style_perm)
        content_ = results["content"]
        style_ = results["style"]
        kl_ = results["KL"]

        outputs, sv = outputs
        latent = sv["latent"]
        intra_f, intra_z = intra_latent_dist(*latent)
        intra_content, intra_style = self.vae_perceptual.perceptual.intra_content_style_loss(style_perm=style_perm)

        # from inference import _latent
        # c_idxs, id = torch.sort(cluster_idxs, stable=True)
        # rperm = torch.argsort(id)[style_perm[id]]
        # _latent(intra_f[:, id][id, :], intra_z[:, id][id, :], intra_content[:, id][id, :], intra_style[:, id][id, :], c_idxs, perm=rperm)

        def select(x):
            mask = torch.triu(torch.ones_like(x) == 1, 1)
            return x[mask]

        intra_f = select(intra_f)
        intra_z = select(intra_z)
        intra_content = select(intra_content)
        intra_style = select(intra_style)

        # todo: Spearman
        corr_content = torch.corrcoef(torch.stack([intra_z, intra_content]))[0, 1]
        corr_style = torch.corrcoef(torch.stack([intra_f, intra_style]))[0, 1]
        corr_dis = 1 - 0.5 * (corr_content + corr_style)
        if torch.isnan(corr_dis):
            corr_dis = 0
            corr_content = 0
            corr_style = 0

        results.update(
            {"DIS-perceptual": self.l[0] * content_ + self.l[1] * style_ + self.l[2] * corr_dis + self.l[3] * kl_,
             "corr-content": corr_content, "corr-style": corr_style, "corr": corr_dis}
        )
        if cluster_idxs is not None:
            for c in np.unique(cluster_idxs):
                content_ = results["content_%d" % c]
                style_ = results["style_%d" % c]
                kl_ = results["KL_%d" % c]
                results.update(
                    {"DIS-perceptual_%d" % c: self.l[0] * content_ + self.l[1] * style_ + self.l[2] * corr_dis + self.l[3] * kl_})
        return results


class GP_Likelihood_Shift(Metric):
    def __init__(self, noise=0, **kwargs):
        self.kernel = RBF()
        self.name = "GP-Likelihood-shift"
        if noise > 0:
            self.kernel = self.kernel + WhiteKernel(noise)
            self.name = "GP-Likelihood-shift-noise"

    @vae_compatible_call
    def __call__(self, outputs, labels):
        lk_pre = [gp_likelihood(lab, self.kernel) for lab in labels.cpu().numpy()]
        lk_post = [gp_likelihood(lab, self.kernel) for lab in outputs.cpu().numpy()]
        return {self.name: np.mean([post - pre for (pre, post) in zip(lk_pre, lk_post)])}


# inference-time


def prd(batch1, batch2, num_clusters=6, name=None, t_SNE=False):

    b1 = torch.cat(batch1).detach().cpu()
    b2 = torch.cat(batch2).detach().cpu()
    # b1 = b1 / b1.std(axis=2, keepdim=True)
    # b2 = b2 / b2.std(axis=2, keepdim=True)
    b1 = b1.reshape(b1.shape[0], -1)
    b2 = b2.reshape(b2.shape[0], -1)
    prd = compute_prd_from_embedding(b1, b2, num_clusters=num_clusters)

    if t_SNE:
        from plot import dimred
        pass  # dimred(b1, b2, name=name)

    beta = np.max(prd[1])
    beta_1 = np.max(prd[1][prd[0] > 0.01]) if np.any(prd[0] > 0.01) else 0
    beta_5 = np.max(prd[1][prd[0] > 0.05]) if np.any(prd[0] > 0.05) else 0
    beta_10 = np.max(prd[1][prd[0] > 0.1]) if np.any(prd[0] > 0.1) else 0
    beta_avg = np.mean(prd[1])
    alpha = np.max(prd[0])
    alpha_1 = np.max(prd[0][prd[1] > 0.01]) if np.any(prd[1] > 0.01) else 0
    alpha_5 = np.max(prd[0][prd[1] > 0.05]) if np.any(prd[1] > 0.05) else 0
    alpha_10 = np.max(prd[0][prd[1] > 0.1]) if np.any(prd[1] > 0.1) else 0
    alpha_avg = np.mean(prd[0])
    f = 2 / (1 / beta + 1 / alpha) if min(alpha, beta) > 0 else 0
    f_1 = 2 / (1 / beta_1 + 1 / alpha_1) if min(alpha_1, beta_1) > 0 else 0
    f_5 = 2 / (1 / beta_5 + 1 / alpha_5) if min(alpha_5, beta_5) > 0 else 0
    f_10 = 2 / (1 / beta_10 + 1 / alpha_10) if min(alpha_10, beta_10) > 0 else 0
    f_avg = 2 / (1 / beta_avg + 1 / alpha_avg) if min(alpha_avg, beta_avg) > 0 else 0

    # print('Recall %.4f' % np.max(prd[1]))
    # print('Precision %.4f' % np.max(prd[0]))
    # print("F-score %.4f" % (1 / np.max(1 / np.max(prd, axis=1))))
    out = {
        "recall": beta,
        "recall-1": beta_1,
        "recall-5": beta_5,
        "recall-10": beta_10,
        "recall-avg": beta_avg,
        "precision": alpha,
        "precision-1": alpha_1,
        "precision-5": alpha_5,
        "precision-10": alpha_10,
        "precision-avg": alpha_avg,
        "F": f,
        "F-1": f_1,
        "F-5": f_5,
        "F-10": f_10,
        "F-avg": f_avg,
    }
    # Test mode
    if name is None:
        return out
    else:
        return out, prd


# Vanilla LSTM
class LSTM(torch.nn.Module):
    def __init__(self, input_size, output_size, input_len, output_len, hidden_size=100):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.1)
        self.linear = torch.nn.Linear(hidden_size, output_size * output_len)
        self.out_shape = (output_len, output_size)

    def forward(self, x, hidden=None):
        lstm_out, self.hidden = self.lstm(x, hidden)
        predictions = self.linear(lstm_out[:, -1]).reshape(x.shape[0], *self.out_shape)
        return predictions


def predictive_utility(batch1, batch2, epochs=10, in_len=50, out_len=5, lr=0.01):
    b, l, c = batch1[0].shape
    net = LSTM(c, c, in_len, out_len).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    mae = torch.nn.L1Loss()

    net.train()
    losses = []
    history = []
    done = False
    for epoch in range(1, epochs + 1):
        for ts in batch1:
            xs = []
            ys = []
            for st in range(0, l - in_len - out_len):
                xs += [ts[:, st:(st + in_len)]]
                ys += [ts[:, (st + in_len):(st + in_len + out_len)]]
            xs = torch.cat(xs)
            ys = torch.cat(ys)
            for x, y in zip(torch.split(xs, 200), torch.split(ys, 200)):
                x = x.to(device)
                y = y.to(device)
                y_hat = net(x, None)
                optimizer.zero_grad()
                loss = criterion(y_hat, y)
                loss.backward()
                losses += [loss.item()]
                # 15th 50-iterations-block without improvements
                if len(losses) > 500:
                    history += [np.mean(losses[-500:])]
                    if len(history) > 15 and epoch > 1 and history[-15] == np.min(history[-15:]):
                        done = True
                        break
                optimizer.step()
            if done:
                break
        if epoch % 1 == 0:
            print('[%d/%d]: %.4f' % (epoch, epochs, history[-1] if len(history) > 0 else np.mean(losses)))
        if done:
            break

    net.eval()
    # mses, maes = [], []
    # with torch.no_grad():
    #     for ts in batch1:
    #         xs = []
    #         ys = []
    #         for st in range(0, l - in_len - out_len):
    #             xs += [ts[:, st:(st + in_len)]]
    #             ys += [ts[:, (st + in_len):(st + in_len + out_len)]]
    #         xs = torch.cat(xs)
    #         ys = torch.cat(ys)
    #         for x, y in zip(xs, ys):
    #             x = x.to(device)
    #             y = y.to(device)
    #             y_hat = net(x, None)
    #             mses += [mse(y_hat, y).item()]
    #             maes += [mae(y_hat, y).item()]
    # print("MAE: %.6f" % np.mean(maes))
    # print("MSE: %.6f" % np.mean(mses))

    mses, maes = [], []
    with torch.no_grad():
        for ts in batch2:
            xs = []
            ys = []
            for st in range(0, l - in_len - out_len):
                xs += [ts[:, st:(st + in_len)]]
                ys += [ts[:, (st + in_len):(st + in_len + out_len)]]
            xs = torch.cat(xs)
            ys = torch.cat(ys)
            for x, y in zip(torch.split(xs, 1000), torch.split(ys, 1000)):
                x = x.to(device)
                y = y.to(device)
                y_hat = net(x, None)
                mses += [mse(y_hat, y).item()]
                maes += [mae(y_hat, y).item()]
    # print("MAE: %.6f" % np.mean(maes))
    # print("MSE: %.6f" % np.mean(mses))
    # todo: last batch smaller
    out = {"p-MAE": np.mean(maes), "p-MSE": np.mean(mses)}
    return out


# Vanilla CNN classifier
class CNN(torch.nn.Module):
    def __init__(self, in_len, input_size, conv_channels=25):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(input_size, conv_channels, 5)
        self.conv2 = torch.nn.Conv1d(conv_channels, 2 * conv_channels, 5)
        self.pool = torch.nn.MaxPool1d(2, 2)
        self.fc1 = torch.nn.Linear(2 * conv_channels * ((in_len - 4) // 2 - 4) // 2, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = x.swapaxes(1, 2)
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# LR
class LR(torch.nn.Module):
    def __init__(self, in_len, input_size, conv_channels=25):
        super().__init__()
        self.fc = torch.nn.Linear(in_len * input_size, 1)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


def classify_style(batch1, batch2, epochs=10, lr=0.01):
    y1 = [torch.zeros(x.shape[0], 1) for x in batch1]
    y2 = [torch.ones(x.shape[0], 1) for x in batch2]
    vals = int(len(batch1) * 0.5)
    b1, b1_v = batch1[:vals], batch1[vals:]
    b2, b2_v = batch2[:vals], batch2[vals:]
    y1, y1_v = y1[:vals], y1[vals:]
    y2, y2_v = y2[:vals], y2[vals:]

    b, l, c = batch1[0].shape
    net = LR(l, c).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    net.train()
    losses = []
    history = []
    done = False
    for epoch in range(1, epochs + 1):
        for bb1, bb2, yy1, yy2 in zip(b1, b2, y1, y2):
            xs = torch.cat([bb1, bb2])
            ys = torch.cat([yy1, yy2])
            r = torch.randperm(len(xs))
            xs = xs[r]
            ys = ys[r]
            for x, y in zip(torch.split(xs, 200), torch.split(ys, 200)):
                x = x.to(device)
                y = y.to(device)
                y_hat = net(x)
                optimizer.zero_grad()
                loss = criterion(y_hat, y)
                loss.backward()
                losses += [loss.item()]
                # 15th 50-iterations-block without improvements
                if len(losses) > 200:
                    history += [np.mean(losses[-200:])]
                    if len(history) > 15 and epoch > 1 and history[-15] == np.min(history[-15:]):
                        done = True
                        break
                optimizer.step()
            if done:
                break
        if epoch % 1 == 0:
            print('[%d/%d]: %.4f' % (epoch, epochs, history[-1] if len(history) > 0 else np.mean(losses)))
        if done:
            break

    net.eval()
    conf = np.zeros((2, 2))
    with torch.no_grad():
        for xs, ys in zip(b1_v + b2_v, y1_v + y2_v):
            for x, y in zip(torch.split(xs, 200), torch.split(ys, 200)):
                x = x.to(device)
                y = y.to(device)
                y_hat = torch.round(torch.sigmoid(net(x)))
                i = int(y[0, 0].cpu())
                conf[i] = conf[i] + np.array([(y_hat == 0).sum().cpu(), (y_hat == 1).sum().cpu()])

    acc = (100 * (conf[0, 0] + conf[1, 1]) / np.sum(conf))
    pre = (100 * conf[1, 1] / np.sum(conf[:, 1]))
    rec = (100 * conf[1, 1] / np.sum(conf[1, :]))
    # print("Confusion matrix:")
    # print(conf)
    # print("Acuraccy: %.2f" % acc)
    # print("Precison: %.2f" % pre)
    # print("Recall: %.2f" % rec)
    out = {
        "c-tn": conf[0, 0], "c-fp": conf[0, 1], "c-fn": conf[1, 0], "c-tp": conf[1, 1],
        "c-acc": acc, "c-precision": pre, "c-recall": rec
    }
    # todo: last batch smaller
    return out


# List of available losses / metrics
metric_dict = {
    # batches
    "MSE": MSE_Loss,
    "KL": KL_Loss,
    "perceptual": Perceptual_Loss,
    "VAE-MSE": VAE_MSE_Loss,
    "VAE-perceptual": VAE_Perceptual_Loss,
    "DIS-perceptual": VAE_Perceptual_Disentangled_Loss,
    "GP-Likelihood-shift": partial(GP_Likelihood_Shift, noise=0),
    "GP-Likelihood-shift-noise": partial(GP_Likelihood_Shift, noise=0.5),
    "fin": Finance_Loss,
    "cov": Cov_Loss,
    # datasets
    "PR": prd,
    "MAE": predictive_utility,
    "ACC": classify_style
}
