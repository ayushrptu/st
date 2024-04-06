import torch
import torch.nn as nn

from .autoencoder_blocks import EmbeddedAEBase, VAEBase

"""
1D CNN for Image Classification
"""


class CNNAE(EmbeddedAEBase):
    def __init__(self, in_channels=1, layers=4, out_conv_channels=None, masking="none", kernel_size=4, dropout=0,
                 last_nonempty=None, **kwargs):
        """
        Define an Autoencoder CNN with <layer> blocks of convolution + deconvolution

        :param in_channels: (color) channels of the image, default: 1 (grayscale)
        :param layers: number of convolutional layers, default: 4
        :param out_conv_channels: number of conv. channels in the last layer, default: 16 * 2**layers
        """
        assert masking in ["none", "token-pre"]
        kwargs.update({"embedding": "none", "masking": masking, "positional_encoding": False,
                       "in_channels": in_channels, "out_channels": in_channels,
                       "d_model": in_channels, "final_layer": False})
        super().__init__(**kwargs)
        # use channel sizes [32, 64, 128, ...] and outdim ~ dim/16
        if out_conv_channels is None:
            out_conv_channels = 2 ** (4 + layers)

        self.layers = layers
        self.last_nonempty = last_nonempty
        # negative means reduce
        conv_channels = [in_channels] + [int(out_conv_channels / 2**k) for k in range(self.layers - 1, -1, -1)]
        deconv_channels = conv_channels[::-1]

        # Convolutions with Batch Normalization
        # conv_outdim = dim/2 + (4 - kern)/2
        # deconv_dim = dim*2 + (4 - kern)
        for i in range(self.layers):
            if self.last_nonempty is not None and i > self.last_nonempty:
                break
            conv = nn.Sequential(
                nn.Conv1d(
                    in_channels=conv_channels[i], out_channels=conv_channels[i+1],
                    kernel_size=kernel_size, padding=1, stride=2, bias=False
                ),
                # nn.BatchNorm1d(conv_channels[i+1]),
                nn.Dropout(dropout),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.add_module("conv%d" % i, conv)
            # if self.last_nonempty is None:
            #     deconv = nn.Sequential(
            #         nn.ConvTranspose1d(
            #             in_channels=deconv_channels[i], out_channels=deconv_channels[i+1],
            #             kernel_size=kernel_size, padding=1, stride=2, bias=False
            #         ),
            #         # nn.BatchNorm1d(deconv_channels[i+1]),
            #         nn.LeakyReLU(0.2, inplace=True)
            #     )
            #     self.add_module("deconv%d" % i, deconv)

        for i in range(self.layers):
            # if self.last_nonempty is not None and i > self.last_nonempty:
            #     break
            # conv = nn.Sequential(
            #     nn.Conv1d(
            #         in_channels=conv_channels[i], out_channels=conv_channels[i+1],
            #         kernel_size=kernel_size, padding=1, stride=2, bias=False
            #     ),
            #     # nn.BatchNorm1d(conv_channels[i+1]),
            #     nn.Dropout(dropout),
            #     nn.LeakyReLU(0.2, inplace=True)
            # )
            # self.add_module("conv%d" % i, conv)
            if self.last_nonempty is None:
                deconv = nn.Sequential(
                    nn.ConvTranspose1d(
                        in_channels=deconv_channels[i], out_channels=deconv_channels[i+1],
                        kernel_size=kernel_size, padding=1, stride=2, bias=False
                    ),
                    # nn.BatchNorm1d(deconv_channels[i+1]),
                    nn.LeakyReLU(0.2, inplace=True)
                )
                self.add_module("deconv%d" % i, deconv)

    def transform(self, x):
        x = x.permute(0, 2, 1)
        for i in range(self.layers):
            conv = getattr(self, "conv%d" % i)
            x = conv(x)
            if i == self.last_nonempty:
                return x
        for i in range(self.layers):
            deconv = getattr(self, "deconv%d" % i)
            x = deconv(x)
        x = x.permute(0, 2, 1)
        return x


class CNNAE2(EmbeddedAEBase):
    def __init__(self, in_channels=1, layers=4, out_conv_channels=None, masking="none", kernel_size=4, dropout=0,
                 last_nonempty=None, **kwargs):
        """
        Define an Autoencoder CNN with <layer> blocks of convolution + deconvolution

        :param in_channels: (color) channels of the image, default: 1 (grayscale)
        :param layers: number of convolutional layers, default: 4
        :param out_conv_channels: number of conv. channels in the last layer, default: 16 * 2**layers
        """
        assert masking in ["none", "token-pre"]
        kwargs.update({"embedding": "none", "masking": masking, "positional_encoding": False,
                       "in_channels": in_channels, "out_channels": in_channels,
                       "d_model": in_channels, "final_layer": False})
        super().__init__(**kwargs)
        # use channel sizes [32, 64, 128, ...] and outdim ~ dim/16
        if out_conv_channels is None:
            out_conv_channels = 2 ** (4 + layers)

        self.layers = layers
        self.last_nonempty = last_nonempty
        # negative means reduce
        conv_channels = [1] + [int(out_conv_channels / 2**k) for k in range(self.layers - 1, -1, -1)]
        deconv_channels = conv_channels[::-1]

        # Convolutions with Batch Normalization
        # conv_outdim = dim/2 + (4 - kern)/2
        # deconv_dim = dim*2 + (4 - kern)
        for i in range(self.layers):
            if self.last_nonempty is not None and i > self.last_nonempty:
                break
            conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=conv_channels[i], out_channels=conv_channels[i+1],
                    kernel_size=kernel_size, padding=1, stride=2, bias=False
                ),
                # nn.BatchNorm1d(conv_channels[i+1]),
                nn.Dropout(dropout),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.add_module("conv%d" % i, conv)
            if self.last_nonempty is None:
                deconv = nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=deconv_channels[i], out_channels=deconv_channels[i+1],
                        kernel_size=kernel_size, padding=1, stride=2, bias=False
                    ),
                    # nn.BatchNorm1d(deconv_channels[i+1]),
                    nn.LeakyReLU(0.2, inplace=True)
                )
                self.add_module("deconv%d" % i, deconv)

    def transform(self, x):
        x = x.unsqueeze(1)
        for i in range(self.layers):
            conv = getattr(self, "conv%d" % i)
            x = conv(x)
            if i == self.last_nonempty:
                return x
        for i in range(self.layers):
            deconv = getattr(self, "deconv%d" % i)
            x = deconv(x)
        x = torch.nn.functional.pad(x, (1, 0), mode="constant", value=float(x.mean()))
        x = x.squeeze(1)
        return x


class CNN_(torch.nn.Module):

    def __init__(self, channels, *args, layers=2, kernel_size=5, dropout=0.1, reduction="mean", **kwargs):
        super().__init__()
        self.layers = layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.in_channels = channels
        for i in range(self.layers):
            conv = nn.Sequential(
                nn.Conv1d(
                    in_channels=self.in_channels, out_channels=self.in_channels,
                    kernel_size=kernel_size, padding="same"
                ),
                nn.Dropout(dropout),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.add_module("conv%d" % i, conv)

    def forward(self, x, *args, **kwargs):
        x = x.swapaxes(1, 2)
        for i in range(self.layers):
            conv = getattr(self, "conv%d" % i)
            x = conv(x)
        x = x.swapaxes(2, 1)
        return x


# MA-CNN
class CNN_Disentangled(VAEBase):
    def __init__(self, *args, is_vae=False, layers=2, d_latent_style=64, kernel_size=5,
                 pe_after_latent=False, positional_encoding=False,
                 dropout=0.1, mode="factorised", **kwargs):
        """
        Several bidirectional transformer models, for disentangled latent space representation
        can be trained on source and target style data.

        x -[style_enncoder]-> f
        x (| f) -[content_encoder]-> z
        z | f -[ddecoder]-> x'

        Can be set/unset to a VAE with kwarg "is_vae"

        :param d_model: The dimensions of your model, defaults to 64
        :param mode: "factorised": no dependency between z and f, i.e. style and content are fully separable
                     "full": z depends on f, i.e. style may affect content
        """
        super().__init__(*args, is_vae=is_vae, pe_after_latent=False, positional_encoding=False, **kwargs)
        assert mode in ["factorised", "full"]
        self.mode = mode
        self.d_latent_style = d_latent_style
        self.kernel_size = kernel_size
        self.layers = layers
        self.dropout = dropout

        self.style_encoder_transformer = CNN_(self.d_model, kernel_size=self.kernel_size, layers=self.layers,
                                              dropout=self.dropout)
        self.content_encoder_transformer = CNN_(self.d_model, kernel_size=self.kernel_size, layers=self.layers,
                                                dropout=self.dropout)

        self.decoder_transformer = CNN_(self.d_model, kernel_size=self.kernel_size, layers=self.layers,
                                        dropout=self.dropout)

        # additional fc for global style in latent space
        self.fc_f_mu = torch.nn.Linear(self.d_model, self.d_latent_style)
        self.fc_f_log_std = torch.nn.Linear(self.d_model, self.d_latent_style)
        if mode == "full":
            # fc to make lower?
            # positional encodding
            self.fc_mu = torch.nn.Linear(self.d_model + self.d_latent_style, self.d_latent)
            self.fc_log_std = torch.nn.Linear(self.d_model + self.d_latent_style, self.d_latent)

        # override with increased latent size
        self.fc_decode = torch.nn.Linear(self.d_latent + self.d_latent_style, self.d_model)

    def transform_encode_f(self, x: torch.Tensor, src_mask=None):
        """ forward without encoding / decoding """
        # query token
        query_token = torch.zeros(x.shape[2]).to(x.device)
        query_token[-2] = 1
        x_query = torch.cat([query_token.reshape(1, 1, -1).expand([x.shape[0], -1, -1]), x], dim=1)
        # extract at query position
        f = self.style_encoder_transformer(x_query, src_mask)
        return f[:, 0:1, :]

    def transform_encode_z(self, x: torch.Tensor, f: torch.Tensor, src_mask=None):
        """ forward without encoding / decoding """
        if self.mode == "full":
            x = torch.cat([x, f], dim=2)
        z = self.content_encoder_transformer(x, src_mask)
        return z

    def transform_decode(self, x: torch.Tensor, src_mask=None):
        """ forward without encoding / decoding """
        x = self.decoder_transformer(x, src_mask)
        return x

    def transform(self, x: torch.Tensor, style=None, style_perm=None, **kwargs):
        """
        pure forward on representation of shape (B, seq_len, d_model)
        optionally, override latent style
        """
        assert self.is_vae
        if style is not None:
            params = style
        else:
            # Encode f
            # Query flag
            x[:, :, -2] = 0
            f = self.transform_encode_f(x, **kwargs)
            params = self.latent_encode(f, self.fc_f_mu, self.fc_f_log_std, perm=style_perm)
        # Resample f
        f = self.dist_q(*params).rsample()
        # Same f for each x_t
        f = f.expand(-1, x.shape[1], -1)
        # Encode z
        z = self.transform_encode_z(x, f, **kwargs)
        # Resample z
        params = self.latent_encode(z, self.fc_mu, self.fc_log_std, keep_kl=True)
        z = self.dist_q(*params).rsample()

        # Decode
        x = torch.cat([z, f], dim=2)
        x = self.latent_decode(x)
        x = self.transform_decode(x, **kwargs)
        return x


if __name__ == "__main__":
    net = CNNAE(2)
    # 345988
    print("%d trainable parameters" % sum([p.numel() for p in net.parameters() if p.requires_grad]))
    inp = torch.zeros(1, 128, 2)
    print(inp.shape, net(inp).shape)
    for name, par in net.named_parameters():
        print(name, par.shape)
