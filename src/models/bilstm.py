from collections import OrderedDict
import torch
from torch.nn.modules import LSTM

from .autoencoder_blocks import EmbeddedAEBase, VAEBase


class BiLSTMAE(EmbeddedAEBase):
    def __init__(self, *args, d_model=128, d_hidden=64, dropout=0.1, layers=2, **kwargs):
        """
        A bidirectional LSTM model

        :param d_model: The dimensions of your model, defaults to 128
        :param d_hidden: The dimensions of the two hidden states, defaults to 64
        :param dropout: The fraction of dropout you wish to apply during training, default 0.1
        :param layers: number of LSTM blocks
        """
        super().__init__(*args, d_model=d_model, positional_encoding=False, **kwargs)
        self.d_hidden = d_hidden
        self.layers = layers
        self.lstms = [LSTM(d_model, d_hidden, num_layers=1, dropout=dropout, bidirectional=True, batch_first=True) for _ in range(self.layers)]
        for i, m in enumerate(self.lstms):
            self.add_module("lstm%d" % i, m)

        # Force input_shape == output_shape
        if 2 * d_hidden == d_model:
            self.reduction = "none"
        elif (2 * d_hidden) % d_model == 0:
            self.reduction = "mean"
        else:
            self.reduction = "linear"
            self.linear = torch.nn.Linear(2 * self.d_hidden, self.d_model)

    def transform(self, x: torch.Tensor):
        """
        forward without encoding / decoding
        :param x: of shape(B, seq_len, d_model)
        """
        for lstm in self.lstms:
            x, _ = lstm(x)
            if self.reduction == "mean":
                k = (2 * self.d_hidden) // self.d_model
                x = torch.mean(torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2] // k, k)), 3)
            elif self.reduction == "linear":
                x = self.linear(x)
        return x


class BiLSTMAE_Double(EmbeddedAEBase):
    def __init__(self, *args, d_model=128, d_hidden=64, dropout=0.1, layers=2, **kwargs):
        """
        Two consecutive (pretrained) bidirectional LSTM models,
        can be fine-tuned on data whose style is to be adapted.

        :param d_model: The dimensions of your model, defaults to 128
        :param d_hidden: The dimensions of the two hidden states, defaults to 64
        :param dropout: The fraction of dropout you wish to apply during training, default 0.1
        :param layers: number of LSTM blocks
        """
        super().__init__(*args, d_model=d_model, positional_encoding=False, **kwargs)
        self.d_hidden = d_hidden
        self.layers = layers
        self.encoder_lstms = [LSTM(d_model, d_hidden, num_layers=1, dropout=dropout, bidirectional=True, batch_first=True)
                              for _ in range(self.layers)]
        self.decoder_lstms = [LSTM(d_model, d_hidden, num_layers=1, dropout=dropout, bidirectional=True, batch_first=True)
                              for _ in range(self.layers)]
        for i, (m1, m2) in enumerate(zip(self.encoder_lstms, self.decoder_lstms)):
            self.add_module("encoder_lstm%d" % i, m1)
            self.add_module("decoder_lstm%d" % i, m2)

        # Force input_shape == output_shape
        if 2 * d_hidden == d_model:
            self.reduction = "none"
        elif (2 * d_hidden) % d_model == 0:
            self.reduction = "mean"
        else:
            self.reduction = "linear"
            self.linear = torch.nn.Linear(2 * self.d_hidden, self.d_model)

    def transform(self, x: torch.Tensor):
        """
        forward without encoding / decoding
        :param x: of shape(B, seq_len, d_model)
        """
        for lstm in self.encoder_lstms + self.decoder_lstms:
            x, _ = lstm(x)
            # todo: dropout
            if self.reduction == "mean":
                k = (2 * self.d_hidden) // self.d_model
                x = torch.mean(torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2] // k, k)), 3)
            elif self.reduction == "linear":
                x = self.linear(x)
        return x

    def load_state_dict(self, state_dict, strict=False):
        """ load from a pretrained parameter set of a TransformerAE or TransformerAE_Style class """
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("lstm"):
                new_state_dict[k.replace("lstm", "encoder_lstm")] = v
                new_state_dict[k.replace("lstm", "decoder_lstm")] = v.clone()
            else:
                new_state_dict[k] = v
        super().load_state_dict(new_state_dict, strict=strict)


class LSTM_con(torch.nn.Module):

    def __init__(self, *args, layers=2, reduction="mean", **kwargs):
        super().__init__()
        self.layers = layers
        self.reduction = reduction
        self.lstms = [torch.nn.LSTM(*args, **kwargs) for _ in range(self.layers)]
        for i, m in enumerate(self.lstms):
            self.add_module("lstm%d" % i, m)

    def forward(self, x, *args, **kwargs):
        for i in range(self.layers):
            x, _ = self.lstms[i](x)
            if self.reduction == "mean":
                k = 2
                x = torch.mean(torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2] // k, k)), 3)
            elif self.reduction == "linear":
                x = self.linear(x)
        return x


# MA-LSTM
class BiLSTM_Disentangled(VAEBase):
    def __init__(self, *args, is_vae=False, layers=2, d_latent_style=64,
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
        self.d_hidden = self.d_model
        self.layers = layers
        self.dropout = dropout

        self.style_encoder_transformer = LSTM_con(self.d_model, self.d_hidden, layers=self.layers, batch_first=True,
                                                  dropout=self.dropout, bidirectional=True)
        self.content_encoder_transformer = LSTM_con(self.d_model, self.d_hidden, layers=self.layers, batch_first=True,
                                                    dropout=self.dropout, bidirectional=True)

        self.decoder_transformer = LSTM_con(self.d_model, self.d_hidden, layers=self.layers, batch_first=True,
                                            dropout=self.dropout, bidirectional=True)

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

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        super().load_state_dict(state_dict, strict=False)


if __name__ == "__main__":
    net = BiLSTMAE(2, final_layer=True)
    # 199.169
    print("%d trainable parameters" % sum([p.numel() for p in net.parameters() if p.requires_grad]))
    inp = torch.zeros(1, 50, 2)
    print(inp.shape, net(inp).shape)
    for name, par in net.named_parameters():
        print(name, par.shape)
