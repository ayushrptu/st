import torch
import numpy as np
from metrics import metric_dict as select_metric
from models import InputOpt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def iterate(fmodel, batch, style_batch, init=None, num_iters=500,
            lr=0.01, lr_full=200, lr_decay=0.99,
            criterion="perceptual", optimizer="Adam", criterion_params=dict()):
    """
    data-based style transfer by GD on backpropagated gradients

    similar to analyze.py but without clutter from animation
    """
    assert optimizer == "Adam"

    net = InputOpt(*batch.shape, init=init, init_mean=batch.mean(axis=1, keepdim=True).cpu()).to(device)
    # net = InputOpt(*batch.shape, init=init, init_mean=None).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    # print("-----------------89-------------------")
    # print("Fmodel", fmodel)
    # print("-----------------89-------------------")
    # print("criteion", criterion)
    # print("-----------------89-------------------")

    # print("criteion", criterion_params)

    loss_func = select_metric[criterion](fmodel, **criterion_params)

   

    for iter_ in range(num_iters):
        # if iter_ % 100 == 0:
        #     print(iter_)
        #     import matplotlib.pyplot as plt
        #     plt.plot(style_batch[0, :, 0].cpu())
        #     plt.plot(batch[0, :, 0].cpu())
        #     plt.plot(net()[0, :, 0].detach().cpu())
        optimizer.zero_grad()
        ts = net()
        loss = loss_func(ts, batch, style_labels=style_batch)[criterion]
        loss.backward()
        optimizer.step()
        if iter_ > lr_full:
            scheduler.step()

    return net().detach().cpu()
