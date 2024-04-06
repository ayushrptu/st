from collections import OrderedDict

import torch
from torch.nn.modules import Transformer, TransformerEncoder, TransformerEncoderLayer, LayerNorm

from .autoencoder_blocks import EmbeddedAEBase, VAEBase
from tools import LambdaModule


# todo: https://arxiv.org/pdf/2002.04745.pdf


class TransformerAEBase(EmbeddedAEBase):
    """
    a baseclass for different transformer architectures, providing encoding/embedding and decoding
    """

    def __init__(self, *args,
                 layers: int = 2,
                 n_heads: int = 8,
                 dim_feedforward: int = 2048,
                 dropout=0.1,
                 **kwargs):
        """
        A baseclass for different transformer architectures, providing encoding/embedding and decoding

        :param d_model: The dimensions of your model, defaults to 128
        :param n_heads: The number of heads in each encoder/decoder block, defaults 2
        :param dim_feedforward: The number of hidden nodes in the pointwise feedforward (d_model -> dim_ff -> d_model)
        :param dropout: The fraction of dropout you wish to apply during training, default 0.1
        :param layers: number of transformer blocks
        """
        super().__init__(*args, **kwargs)
        self.layers = layers
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout


class TransformerVAEBase(VAEBase):
    """
    a baseclass for different variational autoencoder transformer architectures, providing encoding/embedding and decoding
    """

    def __init__(self, *args,
                 layers: int = 2,
                 n_heads: int = 8,
                 dim_feedforward: int = 2048,
                 dropout=0.1,
                 **kwargs):
        """
        A baseclass for different transformer architectures, providing encoding/embedding and decoding

        :param d_model: The dimensions of your model, defaults to 128
        :param n_heads: The number of heads in each encoder/decoder block, defaults 2
        :param dim_feedforward: The number of hidden nodes in the pointwise feedforward (d_model -> dim_ff -> d_model)
        :param dropout: The fraction of dropout you wish to apply during training, default 0.1
        :param layers: number of transformer blocks
        """
        super().__init__(*args, **kwargs)
        self.layers = layers
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout


# todo: remove
class Transformer2AE(TransformerAEBase):
    def __init__(self, *args, **kwargs):
        """
        A full bidirectional transformer model, i.e. BERT
        legacy with batch_first=False
        """
        super().__init__(*args, **kwargs)
        # todo: TransformerEncoderBlock for better readability, batch_first=True
        self.transformer = Transformer(self.d_model, nhead=self.n_heads, dropout=self.dropout,
                                       dim_feedforward=self.dim_feedforward,
                                       num_encoder_layers=self.layers, num_decoder_layers=0)

    def transform(self, x: torch.Tensor, attn_mask=None):
        """ forward without encoding / decoding """
        # No mask in self attention, because bidirectional
        x = x.permute(1, 0, 2)
        x = self.transformer.encoder(x, attn_mask)
        x = x.permute(1, 0, 2)
        return x


class TransformerAE(TransformerAEBase):
    def __init__(self, *args, **kwargs):
        """
        A full bidirectional transformer model, i.e. BERT
        """
        super().__init__(*args, **kwargs)
        # todo: TransformerEncoderBlock for better readability, batch_first=True
        self.transformer = Transformer(self.d_model, nhead=self.n_heads, dropout=self.dropout,
                                       dim_feedforward=self.dim_feedforward,
                                       num_encoder_layers=self.layers, num_decoder_layers=0, batch_first=True,
                                       norm_first=True)

    def transform(self, x: torch.Tensor, attn_mask=None):
        """ forward without encoding / decoding """
        # No mask in self attention, because bidirectional
        x = self.transformer.encoder(x, attn_mask)
        return x


class TransformerAE_Double(TransformerVAEBase):
    def __init__(self, *args, is_vae=False, **kwargs):
        """
        Two consecutive (pretrained) bidirectional transformer models,
        can be fine-tuned on data whose style is to be adapted.

        Can be set/unset to a VAE with kwarg "is_vae"
        """
        super().__init__(*args, is_vae=is_vae, **kwargs)
        # todo: TransformerEncoderBlock for better readability, batch_first=True
        self.encoder_transformer = Transformer(
            self.d_model, nhead=self.n_heads, dropout=self.dropout, dim_feedforward=self.dim_feedforward,
            num_encoder_layers=self.layers, num_decoder_layers=0, batch_first=True, norm_first=True)
        self.decoder_transformer = Transformer(
            self.d_model, nhead=self.n_heads, dropout=self.dropout, dim_feedforward=self.dim_feedforward,
            num_encoder_layers=self.layers, num_decoder_layers=0, batch_first=True, norm_first=True)

    def transform_encode(self, x: torch.Tensor, src_mask=None):
        """ forward without encoding / decoding """
        x = self.encoder_transformer.encoder(x, src_mask)
        return x

    def transform_decode(self, x: torch.Tensor, src_mask=None):
        """ forward without encoding / decoding """
        x = self.decoder_transformer.encoder(x, src_mask)
        return x

    def load_state_dict(self, state_dict, strict=False):
        """ load from a pretrained parameter set of a TransformerAE or TransformerAE_Style class """
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("transformer"):
                new_state_dict[k.replace("transformer", "encoder_transformer")] = v
                new_state_dict[k.replace("transformer", "decoder_transformer")] = v.clone()
            else:
                new_state_dict[k] = v
        super().load_state_dict(new_state_dict, strict=strict)


class TransformerAE_Disentangled(TransformerVAEBase):
    def __init__(self, *args, is_vae=False, d_latent_style=64, mode="factorised", **kwargs):
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
        super().__init__(*args, is_vae=is_vae, **kwargs)
        assert mode in ["factorised", "full"]
        self.mode = mode
        self.d_latent_style = d_latent_style

        # todo: TransformerEncoderBlock for better readability
        self.style_encoder_transformer = Transformer(
            self.d_model, nhead=self.n_heads, dropout=self.dropout, dim_feedforward=self.dim_feedforward,
            num_encoder_layers=self.layers, num_decoder_layers=0, batch_first=True, norm_first=True)
        self.content_encoder_transformer = Transformer(
            self.d_model if mode == "factorised" else self.d_model + self.d_latent_style,
            nhead=self.n_heads, dropout=self.dropout, dim_feedforward=self.dim_feedforward,
            num_encoder_layers=self.layers, num_decoder_layers=0, batch_first=True, norm_first=True)
        self.decoder_transformer = Transformer(
            self.d_model, nhead=self.n_heads, dropout=self.dropout, dim_feedforward=self.dim_feedforward,
            num_encoder_layers=self.layers, num_decoder_layers=0, batch_first=True, norm_first=True)

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
        f = self.style_encoder_transformer.encoder(x_query, src_mask)
        return f[:, 0:1, :]

    def transform_encode_z(self, x: torch.Tensor, f: torch.Tensor, src_mask=None):
        """ forward without encoding / decoding """
        if self.mode == "full":
            x = torch.cat([x, f], dim=2)
        z = self.content_encoder_transformer.encoder(x, src_mask)
        return z

    def transform_decode(self, x: torch.Tensor, src_mask=None):
        """ forward without encoding / decoding """
        x = self.decoder_transformer.encoder(x, src_mask)
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
    net = TransformerAE(2, final_layer=True)
    # 1.187.073
    print("%d trainable parameters" % sum([p.numel() for p in net.parameters() if p.requires_grad]))
    inp = torch.zeros(3, 50, 2)
    print("\n", inp.shape, net(inp).shape, "\n")
    for name, par in net.named_parameters():
        print(name, par.shape)

    # 2.386.113 = 2 * 1.187.073 + 11.967
    net2 = TransformerAE_Double(2, final_layer=True, is_vae=True)
    print("%d trainable parameters" % sum([p.numel() for p in net2.parameters() if p.requires_grad]))
    net2.load_state_dict(net.state_dict(), strict=False)
    inp = torch.zeros(3, 50, 2)
    out, kl = net2(inp)
    print("\n", inp.shape, out.shape, kl.shape, "\n")
    for name, par in net2.named_parameters():
        print(name, par.shape)
