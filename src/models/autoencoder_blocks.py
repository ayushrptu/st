import math
import torch


class EmbeddedAEBase(torch.nn.Module):
    """
    a baseclass for different architectures (e.g. based on transformers),
    providing embedding(encoding) and "de-embedding"(decoding) of time series
    to a representation of shape (B, seq_len, d_model)
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int = 1,
                 d_model: int = 128,
                 positional_encoding=True,
                 embedding="none",
                 masking="none",
                 context_size=None,
                 context_dim=None,
                 context_layers=1,
                 final_layer=True):
        """
        :param in_channels: The total number of parallel time series in input (e.g. n_feature_time_series + n_targets)
        :param out_channels: The total number of time series to predict/reconstruct (e.g. n_targets)
        :param d_model: The size of the representation/model for each position, defaults to 128

        :param positional_encoding: if to encode position into representation (e.g. for transformers)
        :param embedding: "none": no embedding if in_channels = d_model else "linear"
                          "linear": pointwise linear
                          "conv": convolution, then pointwise linear
                          "conv-chunk": split into patches, then patchwise linear (seq_len = input_len / context_size)
        :param masking: "none" / "zero": no additional masking behaviour, i.e. just setting to zero is assumed
                        "token-pre": additional channel with mask is added before embedding, so that mask embedding is learned
                        "token-post": embedded mask positions are set all-zero after embedding, last dimension in embedding reserved for masking flag,

        # additionally, 2nd-to-last dimension may be reserved for query flag and query token may be appended (not here)

        If embedding with a convolution:
        :param context_size: convolutional kernel size
        :param context_dim: amount of convolutional channels
        :param context_layers: amount of convolutional layers

        :param final_layer: if to use a final layer, here linear for regression
        """
        super().__init__()
        assert embedding in ["none", "linear", "conv", "conv-chunk"], "unknown embedding behaviour"
        assert masking in ["none", "zero", "token-pre", "token-post"], "unknown masking behaviour"
        self.embedding = embedding
        self.masking = masking
        self.in_channels = in_channels + (1 if masking == "token-pre" else 0)
        self.pre_embedding_channels = context_dim if embedding in ["conv", "conv-chunk"] else self.in_channels
        self.d_model = d_model
        self.out_channels = out_channels

        self.positional_encoding = positional_encoding
        self.context_size = context_size
        self.context_dim = context_dim
        self.context_layers = context_layers

        # Encode with linear layer + positional encoding
        if self.embedding == "conv":
            if context_layers > 1:
                layers = [[torch.nn.Conv1d(self.in_channels if i == 0 else context_dim, context_dim,
                                           kernel_size=context_size, padding="same", padding_mode="replicate"),
                           torch.nn.ReLU()]
                          for i in range(context_layers - 1)]
                layers = [x for l in layers for x in l] +\
                         [torch.nn.Conv1d(context_dim, context_dim, kernel_size=context_size)]
                self.context = torch.nn.Sequential(*layers)
            else:
                self.context = torch.nn.Conv1d(self.in_channels, context_dim, kernel_size=context_size,
                                               padding="same", padding_mode="replicate")
        if self.embedding == "conv-chunk":
            if context_layers > 1:
                assert context_size >= 4
                layers = [[torch.nn.Conv1d(self.in_channels if i == 0 else context_dim, context_dim,
                                           kernel_size=4, padding="same", padding_mode="replicate"),
                           torch.nn.ReLU()]
                          for i in range(context_layers - 1)]
                layers = [x for l in layers for x in l] +\
                         [torch.nn.Conv1d(context_dim, context_dim, kernel_size=context_size)]
                self.context = torch.nn.Sequential(*layers)
            else:
                self.context = torch.nn.Conv1d(self.in_channels, context_dim, kernel_size=context_size)

        self.linear_embed = not (self.embedding == "none" and self.in_channels == self.d_model)
        if self.linear_embed:
            self.dense_shape = torch.nn.Linear(self.pre_embedding_channels, self.d_model)
        if self.positional_encoding:
            self.pe = SimplePositionalEncoding(self.d_model)

        self.final_layer = final_layer
        if self.final_layer:
            # todo: more layers / deconvolutions
            if self.embedding == "conv-chunk":
                self.last_layer = torch.nn.Linear(d_model, self.out_channels * self.context_size)
            else:
                self.last_layer = torch.nn.Linear(d_model, self.out_channels)

    def forward(self, x: torch.Tensor, src_mask=None, **kwargs):
        """
        :param x: torch.Tensor of shape (B, L, M)
                  where B is the batch size, L is the sequence length and M is the number of channels / time series
        :param src_mask: torch.Tensor of shape (B, L, M), only needed when masking is "token-pre" or "token-post"
        :param kwargs: (optional) additional input to the transform, e.g. attn_mask for transformers
        :return: a tensor of dimension (B, L, output_dim)
        """
        x = self.encode(x, src_mask)
        x = self.transform(x, **kwargs)
        x = self.decode(x)
        return x

    def encode(self, x, src_mask):
        """ Embed into a higher dimensional space and add a positional encoding """
        # reserved token before linear embedding
        if self.masking == "token-pre":
            if src_mask is None:
                x = torch.cat([x, torch.zeros_like(x)[:, :, 0:1]], dim=2)
            else:
                x = torch.cat([x, src_mask.unsqueeze(dim=2)], dim=2)
        # context awareness
        if self.embedding == "conv":
            x = x.permute(0, 2, 1)
            x = self.context(x)
            x = x.permute(0, 2, 1)
        # patches
        # todo: full conv, patches after?
        # todo: mask <-> context on different patch sizes if masking != "token-post"?
        elif self.embedding == "conv-chunk":
            batch_size, seq_len, num_channels = x.shape
            num_chunks = seq_len // self.context_size
            x = x.permute(0, 2, 1)
            x = x.reshape((batch_size, num_channels, num_chunks, self.context_size))
            # chunks to batch
            x = x.permute(0, 2, 1, 3)
            x = x.reshape((-1, num_channels, self.context_size))
            x = self.context(x)
            x = x.reshape((batch_size, num_chunks, self.context_dim))
            src_mask = src_mask[:, ::self.context_size, :]
        # linear embedding
        if self.linear_embed:
            x = self.dense_shape(x)
        # reserved token after linear embedding
        if self.masking == "token-post":
            x[src_mask] = 0
        # positional encoding
        if self.positional_encoding:
            x = self.pe(x)
        # flag for mask in last entry
        if self.masking == "token-post":
            x[:, :, -1] = src_mask.to(x.dtype)
        return x

    def transform(self, x: torch.Tensor, **kwargs):
        """ pure forward on representation of shape (B, seq_len, d_model) without encoding / decoding """
        raise NotImplementedError

    def decode(self, x):
        if self.final_layer:
            if self.embedding == "conv-chunk":
                x = self.last_layer(x)
                x = x.reshape((x.shape[0], -1, self.out_channels))
            else:
                x = self.last_layer(x)
        return x


class VAEBase(EmbeddedAEBase):
    """
    a baseclass for different architectures, providing embedding and "de-embedding"
    to a representation of shape (B, seq_len, d_model).
    splits transformation into encoder and decoder of a VAE with latent shape (B, seq_len, d_latent)
    """

    def __init__(self, *args,
                 is_vae=True,
                 d_latent: int = 32,
                 vae_embedding="none",
                 vae_dist="gaussian",
                 pe_after_latent=False,
                 keep_latent=False,
                 **kwargs):
        """
        :param d_latent: size of latent space for generative sampling
        :param vae_embedding: "none": no embedding if in_channels = d_model else "linear"
                              "linear": pointwise linear
                              "compress": linear, over all points
        :param is_vae: if set false default back to a non-probabilistic auto encoder, for compatibility
        :param pe_after_latent: add additional positional encoding after the latent decode, before the 2nd transformer
        :param keep_latent: store latent for later use
        """
        super().__init__(*args, **kwargs)
        assert vae_embedding in ["none", "linear"], "unknown embedding behaviour for VAE latent space"
        assert vae_dist in ["gaussian"], "unknown VAE prior"
        self.is_vae = is_vae
        self.pe_after_latent = pe_after_latent
        self.keep_latent = keep_latent
        if is_vae:
            self.d_latent = d_latent
            self.vae_embedding = vae_embedding
            self.dist = vae_dist

            if self.dist == "gaussian":
                self.dist_q = torch.distributions.Normal

            if vae_embedding == "none" or vae_embedding == "linear":
                self.fc_mu = torch.nn.Linear(self.d_model, self.d_latent)
                self.fc_log_std = torch.nn.Linear(self.d_model, self.d_latent)

                self.latent_linear = not (self.embedding == "none" and self.d_latent == self.d_model)
                if self.latent_linear:
                    self.fc_decode = torch.nn.Linear(self.d_latent, self.d_model)
            # todo: compress over all positions?

            # Store KL Loss
            self.kl = None
            self.latent = None

    def latent_encode(self, x, fc_mu=None, fc_log_std=None, keep_kl=False, perm=None):
        """ compute parameters from representation """
        # For compatibility
        if fc_mu is None:
            fc_mu = self.fc_mu
        if fc_log_std is None:
            fc_log_std = self.fc_log_std

        if self.dist == "gaussian":
            mu = fc_mu(x)
            log_std = fc_log_std(x)
            std = torch.exp(log_std)
            if perm is not None:
                mu = mu[perm]
                std = std[perm]

            # Store latent
            if self.keep_latent:
                if not keep_kl:
                    self.latent = None
                if self.latent is None:
                    self.latent = [mu, std]
                else:
                    self.latent += [mu, std]

            # iid: self.kl = c
            # from: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
            # for MVG: https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
            # todo: correlation ~ distance MVG

            ths_kl = 0.5 * (std**2 + mu**2) - log_std - 0.5
            # sequence innvariant = global
            if ths_kl.shape[1] == 1:
                ths_kl = torch.swapaxes(ths_kl, 1, 2)
            ths_kl = torch.sum(ths_kl, axis=tuple(range(2, len(mu.shape))))

            # Store kl
            if not keep_kl:
                self.kl = None
            if self.kl is None:
                self.kl = ths_kl
            else:
                self.kl = torch.cat([self.kl, ths_kl], dim=1)

            return mu, std
        else:
            raise ValueError("Unknown distribution")

    def latent_decode(self, x):
        """ compute representation from sample """
        if self.latent_linear:
            x = self.fc_decode(x)
        if self.pe_after_latent:
            x = self.pe(x)
        return x

    def transform_encode(self, x: torch.Tensor, **kwargs):
        """
        pure forward on representation,
        fc is used hereafter to extract latent parameters (e.g. mean, std of gaussian)
        """
        raise NotImplementedError

    def transform_decode(self, x: torch.Tensor, **kwargs):
        """
        pure forward on representation,
        fc is used hereafter to extract time series again
        """
        raise NotImplementedError

    def transform(self, x: torch.Tensor, **kwargs):
        """ pure forward on representation of shape (B, seq_len, d_model) """
        # Encode
        x = self.transform_encode(x, **kwargs)
        # Resample
        if self.is_vae:
            params = self.latent_encode(x)
            if self.training:
                x = self.dist_q(*params).rsample()
            else:
                # use mean during inference
                x = params[0]
            x = self.latent_decode(x)
        # Decode
        x = self.transform_decode(x, **kwargs)
        return x

    def forward(self, x: torch.Tensor, src_mask=None, **kwargs):
        """
        :param x: torch.Tensor of shape (B, L, M)
                  where B is the batch size, L is the sequence length and M is the number of channels / time series
        :param src_mask: torch.Tensor of shape (B, L, M), only needed when masking is "token-pre" or "token-post"
        :param kwargs: (optional) additional input to the transform, e.g. attn_mask for transformers
        :return: a tensor of dimension (B, L, output_dim)
        """
        x = self.encode(x, src_mask)
        x = self.transform(x, **kwargs)
        x = self.decode(x)
        if self.is_vae:
            sv = {"kl": self.kl}
            if self.keep_latent:
                sv["latent"] = self.latent
            return x, sv
        else:
            return x


class SimplePositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Creates a basic positional encoding"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """
    Generates a square mask for the sequence, only leaving left context unmasked.
    The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    pe = SimplePositionalEncoding(128)
    plt.imshow(pe.pe[:256, 0, :])
    plt.show()

