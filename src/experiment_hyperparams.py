# Hyperparameters of some past runs to avoid clutter

def copy_update(d, **kwargs):
    out = dict(d)
    out.update(**kwargs)
    return out

style = "energy_norm_single.csv"

content = "gp_single.csv"

GP = dict(cloze_len=8, cloze_perc=0.2, seq_len=256, batch_size=64)  # file_name="toy_data.csv"

VAE = dict(model="BERT_VAE", layers=2, d_model=128, d_latent=4, dim_feedforward=512,
           masking="token-post", embedding="conv", context_size=16,
           loss="VAE-perceptual", lr=0.00015, **GP)

VAE_CS = VAE | dict(feature_model="same", feature_load_artifact="bert-tiny-yern2jlv:v19")

CNN = dict(model="CNN", cloze_len=8, cloze_perc=0.2, seq_len=256,  # file_name="toy_data.csv",
           layers=4, out_conv_channels=32 * 2 ** 4,
           masking="token-pre", embedding="none", loss="MSE", lr=0.00015)

CNN_f = dict(model="CNN", cloze_len=8, cloze_perc=0.2, seq_len=256,  # file_name="toy_data.csv",
             layers=4, out_conv_channels=16 * 2 ** 8, last_nonempty=1,
             masking="token-pre", embedding="none", loss="MSE", lr=0.00015)

CNN_CS = dict(model="CNN", cloze_len=8, cloze_perc=0.2, seq_len=256,  # file_name="toy_data.csv",
              layers=4, out_conv_channels=32 * 2 ** 4,
              masking="token-pre", embedding="none", lr=0.00015,
              loss="perceptual")

VAE_DIR_SM = dict(**VAE, dataset="CSV", file_name="toy_data.csv", style_loss="mean_std", style_dataset=True,
                  style_dataset_params=dict(**GP, dataset="CSV", file_name="toy_data_noise_0_5.csv"),
                  feature_model=True, feature_model_params=CNN, feature_load_artifact="bert-tiny-3mvy1523:v19",
                  content_hooks=["conv0.0"], style_hooks=["conv1.0"], last_hook="conv1.0")

VAE_DIR_SP = dict(**VAE, dataset="CSV", file_name="toy_data_noise_0_5.csv", style_loss="mean_std", style_dataset=True,
                  style_dataset_params=dict(**GP, dataset="CSV", file_name="toy_data.csv"),
                  feature_model=True, feature_model_params=CNN, feature_load_artifact="bert-tiny-3mvy1523:v19",
                  content_hooks=["conv0.0"], style_hooks=["conv1.0"], last_hook="conv1.0")

VAE_DIST_OLD = dict(model="BERT_Disentangled", layers=2, d_model=128, dim_feedforward=512,
                    masking="token-post", embedding="conv", context_size=16, loss="VAE-perceptual", lr=0.00015, **GP)

VAE_DIST = dict(model="BERT_Disentangled", layers=2, d_model=128, dim_feedforward=512,
                masking="token-post", embedding="conv", context_size=16, loss="DIS-perceptual", lr=0.00015, **GP)


# ----------------------------------------------------------------------------------------------------------------------

# Misc
# METRICS = dict(metrics=["stval_PR", "stval_fin", "stval_perceptual", "stval_cov"])
METRICS = dict(metrics=["stval_PR", "stval_fin", "stval_perceptual", "stval_MAE"])
# TEST = dict(metrics=["sttest_PR", "sttest_MAE", "sttest_ACC", "sttest_perceptual", "sttest_fin", "sttest_cov"])
TEST = dict(metrics=["sttest_PR", "sttest_MAE", "sttest_ACC", "sttest_perceptual", "sttest_fin"])
NOISE = dict(transform="masking", cloze_len=8, cloze_perc=0.2, mask_rand_none_split=[0.8, 0.1, 0.1])
LAMBS = dict(lamb_content=1, lamb_style=10, lamb_dis=0)

# Models
TRANS = dict(
    # layers=2, dim_feedforward=512, d_model=128,
    n_heads=8, dropout=0.1, masking="token-post", embedding="conv", context_size=16, context_dim=16, context_layers=1,
)
TCNN = dict(
    # layers=2, d_model=128,
    dropout=0.1, masking="token-post", embedding="conv", context_size=16, context_dim=16, context_layers=1,
)
DIS = dict(
    mode="factorised", vae_dist="gaussian", pe_after_latent=True,
    # d_latent_style=64, d_latent=32
)

# --------------------------------------------------------------------

# GP Data
# GP = dict(seq_len=256, batch_size=64, relevant_cols=["x"], transform="none")
GP = dict(seq_len=256, batch_size=64, relevant_cols=["x_1"], transform="none")
# GP = dict(seq_len=256, batch_size=64, relevant_cols=["x_%d" % i for i in range(5)], transform="none")

# without noise
# GP_smooth = dict(**GP, dataset="CSV", file_name="toy_data_big.csv")
# GP_spiky = dict(**GP, dataset="CSV", file_name="toy_data_noise_big.csv")
# GP_both = dict(**GP, dataset="CSVs", file_names=["toy_data_big.csv", "toy_data_noise_big.csv"], csv_weights=[1, 1])
GP_smooth = dict(**GP, dataset="CSV", file_name=content)
GP_spiky = dict(**GP, dataset="CSV", file_name=style)
GP_both = dict(**GP, dataset="CSVs", file_names=[
               content, style], csv_weights=[1, 1])
GP_sm2sp = dict(**GP_smooth, style_dataset=True, style_dataset_params=GP_spiky)
GP_sp2sm = dict(**GP_spiky, style_dataset=True, style_dataset_params=GP_smooth)
# with noise
GP_smooth_d = copy_update(GP_smooth, **NOISE)
GP_spiky_d = copy_update(GP_spiky, **NOISE)
GP_both_d = copy_update(GP_both, **NOISE)
GP_sm2sp_d = copy_update(GP_sm2sp, **NOISE)
GP_sp2sm_d = copy_update(GP_sp2sm, **NOISE)
# for style transfer
GP_iter_sm2sp = dict(**GP, st_dataset=True, st_content_params=GP_smooth, st_style_params=GP_spiky, **METRICS)
GP_iter_sp2sm = dict(**GP, st_dataset=True, st_content_params=GP_spiky, st_style_params=GP_smooth, **METRICS)


# GP features
# GP_feat_train = dict(
#     **GP_both_d, model="CNN", layers=3, out_conv_channels=4 * 2 ** 8, dropout=0.2,
#     masking="token-pre", embedding="none"
# )
out_conv_channels = 4096
layers = 8
feature_load_artifact = "bert-tiny-eecq0ads:latest"
GP_feat_train = dict(
    **GP_both_d, model="CNN", layers=layers, out_conv_channels=out_conv_channels, dropout=0.2,
    masking="token-pre", embedding="none"
)

GP_feat_ = dict(**GP_feat_train, last_nonempty=None)
GP_FEAT = dict(
    feature_model=True, feature_model_params=GP_feat_, feature_load_artifact=feature_load_artifact,  # "bert-tiny-2mbpqg2r:latest",  # "bert-tiny-rkfiwad5:latest",
    content_hooks=["conv0.0"], style_hooks=["conv1.0"], last_hook="conv1.0"
)

# Methods sm->sp
I_DAE = dict(**GP_FEAT, **GP_iter_sm2sp, model="Iter", skip_train=True, dataset="None", st_loss="perceptual")
I_HC = copy_update(I_DAE, st_loss="fin")
M_FT1 = dict(**GP_FEAT, **GP_smooth_d, model="BERT", **TRANS, loss="perceptual")
M_FT2 = copy_update(dict(**GP_FEAT, **GP_iter_sm2sp, model="BERT", **TRANS, loss="perceptual"), **GP_spiky_d)
M_FT2_ = copy_update(dict(**GP_FEAT, **GP_iter_sm2sp, model="BERT", **TRANS, loss="perceptual", style_dataset=True, style_dataset_params=GP_spiky), **GP_smooth_d)
MA_T1 = dict(**GP_FEAT, **GP_both_d, model="BERT_Disentangled", **TRANS, **DIS, loss="DIS-perceptual")
MA_T2 = copy_update(GP_iter_sm2sp, **MA_T1, shuffle_style="half")
MA_LSTM1 = dict(**GP_FEAT, **GP_both_d, model="BiLSTM_Disentangled", **TCNN, **DIS, loss="DIS-perceptual")
MA_LSTM2 = copy_update(GP_iter_sm2sp, **MA_LSTM1, shuffle_style="half")
MA_CNN1 = dict(**GP_FEAT, **GP_both_d, model="CNN_Disentangled", **TCNN, **DIS, loss="DIS-perceptual")
MA_CNN2 = copy_update(GP_iter_sm2sp, **MA_CNN1, shuffle_style="half")

# Methods sp->sm
I_DAE_r = copy_update(I_DAE, **GP_iter_sp2sm)
I_HC_r = copy_update(I_HC, **GP_iter_sp2sm)
M_FT1_r = copy_update(M_FT1, **GP_spiky_d)
M_FT2_r = copy_update(dict(**GP_FEAT, **GP_iter_sp2sm, model="BERT", **TRANS, loss="perceptual"), **GP_smooth_d)
M_FT2_r_ = copy_update(dict(**GP_FEAT, **GP_iter_sp2sm, model="BERT", **TRANS, loss="perceptual", style_dataset=True, style_dataset_params=GP_smooth), **GP_spiky_d)
MA_T2_r = copy_update(GP_iter_sp2sm, **MA_T1, shuffle_style="half")
MA_LSTM2_r = copy_update(GP_iter_sp2sm, **MA_LSTM1, shuffle_style="half")
MA_CNN2_r = copy_update(GP_iter_sp2sm, **MA_CNN1, shuffle_style="half")

# --------------------------------------------------------------------

# Finance data
FIN_ = dict(transform="none", seq_len=256, batch_size=32)
FIN = dict(
    **FIN_, dataset="LabeledCSV", file_name="stock_vol_interp.csv", scale="norm-detrend",
    relevant_cols=['^GSPC', '^IXIC', '^NYA', '^N225', '^HSI'],
)
FIN_S = dict(**FIN_, dataset="FIN", scale="norm-detrend")
FIN_S_n = copy_update(FIN_S, scale="scale-detrend")

# without noise
FIN_low = dict(**FIN, force_same_class=False, select_class=-1)
FIN_high = dict(**FIN, force_same_class=False, select_class=1)
FIN_all = dict(**FIN, force_same_class=False)  # todo: class_weights=[1, 1, 1]
FIN_lo2hi = dict(**FIN_low, style_dataset=True, style_dataset_params=FIN_high)
# with noise
FIN_low_d = copy_update(FIN_low, **NOISE)
FIN_high_d = copy_update(FIN_high, **NOISE)
FIN_all_d = copy_update(FIN_all, **NOISE)
FIN_lo2hi_d = copy_update(FIN_lo2hi, **NOISE)
# without mean correction
FIN_low_n = copy_update(FIN_low, scale="scale-detrend")
FIN_high_n = copy_update(FIN_high, scale="scale-detrend")
FIN_all_n = copy_update(FIN_all, scale="scale-detrend")
FIN_lo2hi_n = copy_update(FIN_lo2hi, scale="scale-detrend")
# without mean correction but with noise
FIN_low_nd = copy_update(FIN_low, scale="scale-detrend", **NOISE)
FIN_high_nd = copy_update(FIN_high, scale="scale-detrend", **NOISE)
FIN_all_nd = copy_update(FIN_all, scale="scale-detrend", **NOISE)
FIN_lo2hi_nd = copy_update(FIN_lo2hi, scale="scale-detrend", **NOISE)
# for style transfer
FIN_iter = dict(**FIN_, st_dataset=True, st_content_params=FIN_low, st_style_params=FIN_high, **METRICS)
FIN_iter_n = dict(**FIN_, st_dataset=True, st_content_params=FIN_low_n, st_style_params=FIN_high_n, **METRICS)

# FIN features
FIN_feat_train = dict(
    **FIN_all, model="CNN", masking="token-pre", embedding="none",
    layers=4, out_conv_channels=16 * 2 ** 8, dropout=0.2
)
FIN_feat_ = dict(**FIN_feat_train, last_nonempty=1)
FIN_FEAT = dict(
    feature_model=True, feature_model_params=FIN_feat_, feature_load_artifact="bert-tiny-8r56lalq:latest",
    content_hooks=["conv0.0"], style_hooks=["conv1.0"], last_hook="conv1.0"
)


# Methods lo->hi
F_I_DAE = dict(**FIN_FEAT, **FIN_iter, model="Iter", skip_train=True, dataset="None", st_loss="perceptual")
F_I_HC = dict(**FIN_FEAT, **FIN_iter_n, model="Iter", skip_train=True, dataset="None", st_loss="fin")
F_MA_T1 = dict(**FIN_FEAT, **FIN_all_d, model="BERT_Disentangled", **TRANS, **DIS, loss="DIS-perceptual")
F_MA_T2 = copy_update(FIN_iter, **F_MA_T1, shuffle_style="half")
F_MA_LSTM1 = dict(**FIN_FEAT, **FIN_all_d, model="BiLSTM_Disentangled", **TCNN, **DIS, loss="DIS-perceptual")
F_MA_LSTM2 = copy_update(FIN_iter, **F_MA_LSTM1, shuffle_style="half")
F_MA_CNN1 = dict(**FIN_FEAT, **FIN_all_d, model="CNN_Disentangled", **TCNN, **DIS, loss="DIS-perceptual")
F_MA_CNN2 = copy_update(FIN_iter, **F_MA_CNN1, shuffle_style="half")
F_M_FT1 = dict(**FIN_FEAT, **FIN_low_d, model="BERT", **TRANS, loss="perceptual")
F_M_FT2_ = copy_update(dict(**FIN_FEAT, **FIN_iter, model="BERT", **TRANS, loss="perceptual", style_dataset=True, style_dataset_params=FIN_high), **FIN_low_d)

# --------------------------------------------------------------------

# TIM Data
TIM_ = dict(seq_len=256, batch_size=16, channels=257, transform="none")
TIM = dict(
    **TIM_, dataset="TIMIT", file_name="TIMIT"
)
TIM_m = dict(**TIM, select_class=1)
TIM_f = dict(**TIM, select_class=0)
TIM_all = dict(**TIM)
TIM_all_d = copy_update(TIM_all, **NOISE)
TIM_iter_m2f = dict(**TIM_, st_dataset=True, st_content_params=TIM_m, st_style_params=TIM_f, **METRICS)
TIM_iter_f2m = dict(**TIM_, st_dataset=True, st_content_params=TIM_f, st_style_params=TIM_m, **METRICS)

# TIM features
TIM_feat_train = dict(
    **TIM_all, model="CNN2", masking="token-pre", embedding="none",
    layers=3, out_conv_channels=8 * 2 ** 8, dropout=0.2
)
TIM_feat_ = dict(**TIM_feat_train, last_nonempty=1)
TIM_FEAT = dict(
    feature_model=True, feature_model_params=TIM_feat_, feature_load_artifact="bert-tiny-2kg9ofyo:latest",
    content_hooks=["conv0.0"], style_hooks=["conv1.0"], last_hook="conv1.0"
)

T_I_DAE = dict(**TIM_FEAT, **TIM_iter_m2f, model="Iter", skip_train=True, dataset="None", st_loss="perceptual")
T_MA_T1 = dict(**TIM_FEAT, **TIM_all_d, model="BERT_Disentangled", **TRANS, **DIS, loss="VAE-perceptual")
T_MA_T2 = copy_update(TIM_iter_m2f, **T_MA_T1, shuffle_style="half")
