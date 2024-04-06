# from .MultiAttnHeadAE import MultiAttnHeadAE
from .transformer import TransformerAE, TransformerAE_Double, TransformerAE_Disentangled
from .bilstm import BiLSTMAE, BiLSTMAE_Double, BiLSTM_Disentangled
from .cnn import CNNAE, CNNAE2, CNN_Disentangled

from .direct_opt import InputOpt

# List of available models
model_dict = {
    # "MultiAttnHeadSimple": MultiAttnHeadAE,
    "Transformer": TransformerAE,
    "Transformer_Style": TransformerAE_Double,
    "Transformer_Disentangled": TransformerAE_Disentangled,
    "BERT": TransformerAE,
    "BERT_Style": TransformerAE_Double,
    "BERT_VAE": TransformerAE_Double,
    "BERT_Disentangled": TransformerAE_Disentangled,
    "BiLSTM_Disentangled": BiLSTM_Disentangled,
    "CNN_Disentangled": CNN_Disentangled,
    "BiLSTM": BiLSTMAE,
    "BiLSTM_Style": BiLSTMAE_Double,
    "CNN": CNNAE,
    "CNN2": CNNAE2,
    "Iter": InputOpt
}
