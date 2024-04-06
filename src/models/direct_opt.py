import matplotlib.pyplot as plt
import torch

# Directly optimize input sequence wrt style loss
# Useful to analyze style features


class InputOpt(torch.nn.Module):
    """
    a baseclass for learning the input directly, without autoencoders
    """
    def __init__(self, batch_size, seq_len, in_channels, init=None, init_mean=None):
        super().__init__()
        if init is None:
            if init_mean is None:
                self.ts = torch.nn.Parameter(2 * torch.rand(batch_size, seq_len, in_channels) - 1, requires_grad=True)
            else:
                self.ts = torch.nn.Parameter(
                    2 * torch.rand(batch_size, seq_len, in_channels) - 1 + init_mean, requires_grad=True
                )
        else:
            self.ts = torch.nn.Parameter(init.detach().cpu(), requires_grad=True)

    def forward(self, *args, **kwargs):
        return self.ts


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    target = torch.rand(1, 32, 1)
    net = InputOpt(1, 32, 1)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
    criterion = torch.nn.MSELoss(reduction="mean")

    fig, ax = plt.subplots()
    writer = PillowWriter(fps=24)

    with writer.saving(fig, "defuse.gif", 100):
        for _ in range(100):
            optimizer.zero_grad()
            ts = net()
            loss = criterion(ts, target)
            loss.backward()
            optimizer.step()

            ax.cla()
            ax.set_ylim(-0.3, 1.3)
            ax.plot(ts[0, :, 0].detach())
            ax.plot(target[0, :, 0].detach())
            writer.grab_frame()
