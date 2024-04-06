import torch

from .autoencoder_blocks import SimplePositionalEncoding

# Code adapted from https://github.com/AIStream-Peelout/flow-forecast


class MultiAttnHeadAE(torch.nn.Module):
    """
    A simple multi-head attention model inspired by Vaswani et al.
    with nidden state of size L x M where L is the sequence length and M is the number of time series
    """

    def __init__(
            self,
            number_time_series: int,
            d_model=128,
            num_heads=8,
            dropout=0.1,
            output_dim=1,
            final_layer=False):

        super().__init__()
        self.dense_shape = torch.nn.Linear(number_time_series, d_model)
        self.pe = SimplePositionalEncoding(d_model)
        self.multi_attn = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)

        self.final_layer = final_layer
        if self.final_layer:
            self.last_layer = torch.nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        :param: x torch.Tensor: of shape (B, L, M)
        Where B is the batch size, L is the sequence length and M is the number of time series
        :return: a tensor of dimension (B, L, output_dim)
        """
        x = self.dense_shape(x)
        x = self.pe(x)
        # Permute to (L, B, M)
        x = x.permute(1, 0, 2)
        if mask is None:
            x = self.multi_attn(x, x, x)[0]
        else:
            x = self.multi_attn(x, x, x, attn_mask=self.mask)[0]
        if self.final_layer:
            x = self.last_layer(x)
        # Switch to (B, L, M)
        x = x.permute(1, 0, 2)
        return x


if __name__ == "__main__":
    net = MultiAttnHeadAE(2, final_layer=True)
    inp = torch.zeros(1, 50, 2)
    print(inp.shape, net(inp).shape)
