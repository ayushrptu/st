from tools import CountedDict

# Store the different configs used

NETS_FOLDER = "../nets/"
DATA_FOLDER = "../data/"
PROJECT = "st_feature_one_channel"


def model_config(valid_perc, test_perc, hyperparams, mode="local"):
    """
    Configure a model by setting hyperparameters
    model: "BERT" / "BERT_Style" / "BERT_VAE"
    dataset: "GP" / "CSV"
    mode: local / wandb / sweep
    """

    hyperparams = CountedDict(hyperparams)

    # training + logging
    assert valid_perc + test_perc < 1
    config = {
        "training_params": {
            "criterion": hyperparams.get("loss", "val_perceptual"),
            "criterion_params": {
                "lamb_content": hyperparams.get("lamb_content", 1),
                "lamb_style": hyperparams.get("lamb_style", 1),
                "lamb_kl": hyperparams.get("lamb_kl", 1),
                "lamb_dis": hyperparams.get("lamb_dis", 0),
                "content_hooks": hyperparams.get("content_hooks", ["context"]),
                "style_hooks": hyperparams.get("style_hooks", None),
                "last_hook": hyperparams.get("last_hook", None),
                "style_loss": hyperparams.get("style_loss", "mean_std")
            },
            "optimizer": "Adam",
            "optim_params": {
                "lr": hyperparams.get("lr", 0.001),
                "lr_decay": hyperparams.get("lr_decay", 1),
                "lr_delay": hyperparams.get("lr_delay", 0),
                "lr_warmup": hyperparams.get("lr_warmup", 0),
                "lr_full": hyperparams.get("lr_full", 0),
            },
            "epochs": hyperparams.get("epochs", 5),
            "shuffle_style": hyperparams.get("shuffle_style", None)
        },
        "skip_train": hyperparams.get("skip_train", False),
        "wandb": mode == "wandb" or mode == "sweep",
        "metrics": hyperparams.get("metrics", ["val_MSE", "test_MSE"]),
        "save_path": NETS_FOLDER + hyperparams.get("name", "bert-tiny")
    }

    # Model
    modelselection_config(config, hyperparams)
    # Features for perceptual loss
    feature_model = hyperparams.get("feature_model", None)
    if feature_model is not None:
        if feature_model == "same":
            config["feature_model"] = config["model"]
            config["feature_model_params"] = config["model_params"]
        else:
            modelselection_config(config, hyperparams["feature_model_params"], prefix="feature_")

    # loading
    assert not ("load_path" in hyperparams.keys() and "load_artifact" in hyperparams.keys())
    if "load_path" in hyperparams.keys():
        config["load_path"] = hyperparams["load_path"]
    if "load_artifact" in hyperparams.keys():
        config["load_artifact"] = hyperparams["load_artifact"]
    if feature_model:
        assert not ("feature_load_path" in hyperparams.keys() and "feature_load_artifact" in hyperparams.keys())
        if "feature_load_path" in hyperparams.keys():
            config["feature_load_path"] = hyperparams["feature_load_path"]
        if "feature_load_artifact" in hyperparams.keys():
            config["feature_load_artifact"] = hyperparams["feature_load_artifact"]

    # Data
    dataselection_config(config, hyperparams, valid_perc, test_perc)
    # style samples
    style_dataset = hyperparams.get("style_dataset", None)
    if style_dataset is not None:
        if style_dataset == "same":
            config["style_dataset"] = config["dataset"]
            config["style_dataset_params"] = config["dataset_params"]
        else:
            dataselection_config(config, hyperparams["style_dataset_params"], valid_perc, test_perc, prefix="style_")
        config["training_params"]["criterion_params"]["style_dataset"] = True
    # style transfer data
    st_dataset = hyperparams.get("st_dataset", None)
    if st_dataset is not None:
        dataselection_config(config, hyperparams["st_content_params"], valid_perc, test_perc, prefix="stc_")
        dataselection_config(config, hyperparams["st_style_params"], valid_perc, test_perc, prefix="sts_")
        config["st_iter_params"] = {
            "criterion": hyperparams.get("st_loss", "perceptual"),
            "criterion_params": hyperparams.get("st_criterion_params", config["training_params"]["criterion_params"]),
            "optimizer": "Adam",
            "lr": hyperparams.get("st_lr", 0.001),
            "lr_full": hyperparams.get("st_lr_full", 20),
            "lr_decay": hyperparams.get("st_lr_decay", 1),
            "num_iters": hyperparams.get("num_iters", 0),
        }

    # Relevant metrics # todo: include criterion by default
    if "perceptual" == config["training_params"]["criterion"]:
        config["metrics"] = config["metrics"] + ["val_perceptual", "test_perceptual"]
    if "VAE-perceptual" == config["training_params"]["criterion"]:
        config["metrics"] = config["metrics"] + ["val_VAE-perceptual", "test_VAE-perceptual"]
    if "DIS-perceptual" == config["training_params"]["criterion"]:
        config["metrics"] = config["metrics"] + ["val_DIS-perceptual", "test_DIS-perceptual"]
        config["model_params"]["keep_latent"] = True

    # Special rules
    if config["model_params"].get("embedding", "none") == "conv-chunk":
        # sequence length permits chunking and chunks align with masking
        assert config["dataset_params"]["seq_len"] % config["model_params"]["context_size"] == 0
        assert config["dataset_params"]["transform_params"]["cloze_len"] == config["model_params"]["context_size"]

    unknown = [k for k, v in hyperparams.items() if hyperparams.counts[k] == 0]
    if len(unknown) > 0:
        raise ValueError("Unknown parameter(s): [%s]" % " ".join(unknown))
    return config


def dataselection_config(config, hyperparams, valid_perc, test_perc, prefix=""):
    """
    Configure the dataset by setting hyperparameters
    dataset: "GP" / "CSV"
    prefix : "" for the training data, "style_" for optional style samples
    """
    # data (+ parameters)
    dataset = hyperparams.get("dataset", "GP")
    if dataset == "GP":
        config.update({
            prefix + "dataset": hyperparams.get("dataset", "GP"),
            prefix + "dataset_params": {
                "n": 5000,

                "valid_perc": valid_perc,
                "test_perc": test_perc,
                "seq_len": hyperparams.get("seq_len", 128),
                "batch_size": hyperparams.get("batch_size", 64),
            }
        })
    elif dataset == "FIN":
        config.update({
            prefix + "dataset": hyperparams.get("dataset", "FIN"),
            prefix + "dataset_params": {
                "n": 320,
                "valid_perc": valid_perc,
                "test_perc": test_perc,
                "seq_len": hyperparams.get("seq_len", 256),
                "batch_size": hyperparams.get("batch_size", 32),

                "scale": hyperparams.get("scale", None)
            }
        })
    elif dataset == "CSV":
        config.update({
            prefix + "dataset": "CSV",
            prefix + "dataset_params": {
                "file_path": DATA_FOLDER + hyperparams.get("file_name"),
                "relevant_cols": hyperparams.get("relevant_cols", ["x"]),
                "target_col": hyperparams.get("target_col", None),

                "valid_perc": valid_perc,
                "test_perc": test_perc,
                "seq_len": hyperparams.get("seq_len", 128),
                "batch_size": hyperparams.get("batch_size", 64),
            },
        })
    elif dataset == "LabeledCSV":
        config.update({
            prefix + "dataset": "LabeledCSV",
            prefix + "dataset_params": {
                "file_path": DATA_FOLDER + hyperparams.get("file_name"),
                "relevant_cols": hyperparams.get("relevant_cols", ["x"]),
                "target_col": hyperparams.get("target_col", None),
                "class_col": hyperparams.get("class_col", "didx"),
                "select_class": hyperparams.get("select_class", None),
                "class_weights": hyperparams.get("class_weights", None),
                "force_same_class": hyperparams.get("force_same_class", True),

                "valid_perc": valid_perc,
                "test_perc": test_perc,
                "seq_len": hyperparams.get("seq_len", 128),
                "batch_size": hyperparams.get("batch_size", 64),

                "scale": hyperparams.get("scale", None)
            },
        })
    elif dataset == "CSVs":
        config.update({
            prefix + "dataset": "CSVs",
            prefix + "dataset_params": {
                "file_paths": [DATA_FOLDER + x for x in hyperparams.get("file_names")],
                "csv_weights": hyperparams.get("csv_weights"),
                "relevant_cols": hyperparams.get("relevant_cols", ["x"]),
                "target_col": hyperparams.get("target_col", None),

                "valid_perc": valid_perc,
                "test_perc": test_perc,
                "seq_len": hyperparams.get("seq_len", 128),
                "batch_size": hyperparams.get("batch_size", 64),
            },
        })
    elif dataset == "TIMIT":
        config.update({
            prefix + "dataset": "TIMIT",
            prefix + "dataset_params": {
                "folder": DATA_FOLDER + hyperparams.get("file_name"),
                "seq_len": hyperparams.get("seq_len", 256),
                "batch_size": hyperparams.get("batch_size", 64),
                "select_class": hyperparams.get("select_class", None),
            },
        })
    elif dataset == "None":
        config[prefix + "dataset"] = None
        config[prefix + "dataset_params"] = None
    else:
        raise ValueError("Dataset not supported")

    # preprocessing
    transform = hyperparams.get("transform", "masking")
    if transform == "masking":
        config[prefix + "dataset_params"].update({
            "transform": "masking",
            "transform_params": {
                "cloze_len": hyperparams.get("cloze_len", 10),
                "cloze_perc": hyperparams.get("cloze_perc", 0.15),
                "mask_rand_none_split": hyperparams.get("mask_rand_none_split", (0.8, 0.1, 0.1))
            }
        })
    elif transform == "gaussian":
        config[prefix + "dataset_params"].update({
            "transform": "gaussian",
            "transform_params": {
                "mean": hyperparams.get("noise_mean", 0),
                "std": hyperparams.get("noise_std", None),
            }
        })
    elif transform == "none":
        pass
    else:
        raise ValueError("Transform not supported")


def modelselection_config(config, hyperparams, prefix=""):
    """
    Configure the model architecture by setting hyperparameters
    model: "BERT" / "BERT_Style" / "BERT_VAE" / "CNN"
    prefix : "" for the model, "feature_" for the feature_model
    """
    if "channels" in hyperparams.keys():
        c_in = hyperparams["channels"]
    elif "relevant_cols" in hyperparams.keys():
        c_in = len(hyperparams["relevant_cols"])
    elif "model_params" in config.keys():
        c_in = config["model_params"]["out_channels"]
    else:
        c_in = 1
    c_out = c_in

    # model (+ parameters)
    model = hyperparams.get("model", "BERT")
    if model == "BERT":
        config.update({
            prefix + "model": "BERT",
            prefix + "model_params": {
                "in_channels": c_in,
                "out_channels": c_out,
                "d_model": hyperparams.get("d_model", 128),
                "layers": hyperparams.get("layers", 2),
                "n_heads": hyperparams.get("n_heads", 8),
                "dim_feedforward": hyperparams.get("dim_feedforward", 2048),
                "dropout": hyperparams.get("dropout", 0.1),
                "embedding": hyperparams.get("embedding", "none"),
                "masking": hyperparams.get("masking", "none"),
                "context_size": hyperparams.get("context_size", 8),
                "context_dim": hyperparams.get("context_dim", 16),
                "context_layers": hyperparams.get("context_layers", 1),
                "final_layer": True,
            },
        })
    elif model == "BERT_Style":
        config.update({
            prefix + "model": "BERT_Style",
            prefix + "model_params": {
                "in_channels": c_in,
                "out_channels": c_out,
                "d_model": hyperparams.get("d_model", 128),
                "layers": hyperparams.get("layers", 2),
                "n_heads": hyperparams.get("n_heads", 8),
                "dim_feedforward": hyperparams.get("dim_feedforward", 2048),
                "dropout": 0.1,
                "embedding": hyperparams.get("embedding", "none"),
                "masking": hyperparams.get("masking", "none"),
                "context_size": hyperparams.get("context_size", 8),
                "context_dim": hyperparams.get("context_dim", 16),
                "context_layers": hyperparams.get("context_layers", 1),
                "final_layer": True,
            },
        })
    elif model == "BERT_VAE":
        config.update({
            prefix + "model": "BERT_VAE",
            prefix + "model_params": {
                "in_channels": c_in,
                "out_channels": c_out,
                "is_vae": True,
                "pe_after_latent": hyperparams.get("pe_after_latent", False),
                "d_model": hyperparams.get("d_model", 128),
                "d_latent": hyperparams.get("d_latent", 32),
                "layers": hyperparams.get("layers", 2),
                "n_heads": hyperparams.get("n_heads", 8),
                "dim_feedforward": hyperparams.get("dim_feedforward", 2048),
                "dropout": 0.1,
                "embedding": hyperparams.get("embedding", "none"),
                "masking": hyperparams.get("masking", "none"),
                "vae_embedding": hyperparams.get("vae_embedding", "none"),
                "vae_dist": hyperparams.get("vae_dist", "gaussian"),
                "context_size": hyperparams.get("context_size", 8),
                "context_dim": hyperparams.get("context_dim", 16),
                "context_layers": hyperparams.get("context_layers", 1),
                "final_layer": True,
            },
        })
    elif model == "BERT_Disentangled":
        config.update({
            prefix + "model": "BERT_Disentangled",
            prefix + "model_params": {
                "in_channels": c_in,
                "out_channels": c_out,
                "is_vae": True,
                "pe_after_latent": hyperparams.get("pe_after_latent", False),
                "mode": hyperparams.get("mode", "factorised"),
                "d_model": hyperparams.get("d_model", 128),
                "d_latent": hyperparams.get("d_latent", 8),
                "d_latent_style": hyperparams.get("d_latent_style", 32),
                "layers": hyperparams.get("layers", 2),
                "n_heads": hyperparams.get("n_heads", 8),
                "dim_feedforward": hyperparams.get("dim_feedforward", 2048),
                "dropout": hyperparams.get("dropout", 0.1),
                "embedding": hyperparams.get("embedding", "none"),
                "masking": hyperparams.get("masking", "none"),
                "vae_embedding": hyperparams.get("vae_embedding", "none"),
                "vae_dist": hyperparams.get("vae_dist", "gaussian"),
                "context_size": hyperparams.get("context_size", 8),
                "context_dim": hyperparams.get("context_dim", 16),
                "context_layers": hyperparams.get("context_layers", 1),
                "final_layer": True,
            },
        })
    elif model == "BiLSTM_Disentangled":
        config.update({
            prefix + "model": "BiLSTM_Disentangled",
            prefix + "model_params": {
                "in_channels": c_in,
                "out_channels": c_out,
                "is_vae": True,
                "pe_after_latent": hyperparams.get("pe_after_latent", False),
                "mode": hyperparams.get("mode", "factorised"),
                "d_model": hyperparams.get("d_model", 128),
                "d_latent": hyperparams.get("d_latent", 8),
                "d_latent_style": hyperparams.get("d_latent_style", 32),
                "layers": hyperparams.get("layers", 2),
                "dropout": hyperparams.get("dropout", 0.1),
                "embedding": hyperparams.get("embedding", "none"),
                "masking": hyperparams.get("masking", "none"),
                "vae_embedding": hyperparams.get("vae_embedding", "none"),
                "vae_dist": hyperparams.get("vae_dist", "gaussian"),
                "context_size": hyperparams.get("context_size", 8),
                "context_dim": hyperparams.get("context_dim", 16),
                "context_layers": hyperparams.get("context_layers", 1),
                "final_layer": True,
            },
        })
    elif model == "CNN_Disentangled":
        config.update({
            prefix + "model": "CNN_Disentangled",
            prefix + "model_params": {
                "in_channels": c_in,
                "out_channels": c_out,
                "kernel_size": hyperparams.get("kernel_size", 128),
                "is_vae": True,
                "positional_encoding": hyperparams.get("positional:encoding", False),
                "pe_after_latent": hyperparams.get("pe_after_latent", False),
                "mode": hyperparams.get("mode", "factorised"),
                "d_model": hyperparams.get("d_model", 128),
                "d_latent": hyperparams.get("d_latent", 8),
                "d_latent_style": hyperparams.get("d_latent_style", 32),
                "layers": hyperparams.get("layers", 2),
                "dropout": hyperparams.get("dropout", 0.1),
                "embedding": hyperparams.get("embedding", "none"),
                "masking": hyperparams.get("masking", "none"),
                "vae_embedding": hyperparams.get("vae_embedding", "none"),
                "vae_dist": hyperparams.get("vae_dist", "gaussian"),
                "context_size": hyperparams.get("context_size", 8),
                "context_dim": hyperparams.get("context_dim", 16),
                "context_layers": hyperparams.get("context_layers", 1),
                "final_layer": True,
            },
        })
    elif model == "BiLSTM":
        config.update({
            prefix + "model": "BiLSTM",
            prefix + "model_params": {
                "in_channels": c_in,
                "out_channels": c_out,
                "d_model": hyperparams.get("d_model", 128),
                "d_hidden": hyperparams.get("d_hidden", 64),
                "layers": hyperparams.get("layers", 2),
                "dropout": 0.1,
                "embedding": hyperparams.get("embedding", "none"),
                "masking": hyperparams.get("masking", "none"),
                "context_size": hyperparams.get("context_size", 8),
                "context_dim": hyperparams.get("context_dim", 16),
                "context_layers": hyperparams.get("context_layers", 1),
                "final_layer": True,
            },
        })
    elif model == "BiLSTM_Style":
        config.update({
            prefix + "model": "BiLSTM_Style",
            prefix + "model_params": {
                "in_channels": c_in,
                "out_channels": c_out,
                "d_model": hyperparams.get("d_model", 128),
                "d_hidden": hyperparams.get("d_hidden", 64),
                "layers": hyperparams.get("layers", 2),
                "dropout": 0.1,
                "embedding": hyperparams.get("embedding", "none"),
                "masking": hyperparams.get("masking", "none"),
                "context_size": hyperparams.get("context_size", 8),
                "context_dim": hyperparams.get("context_dim", 16),
                "context_layers": hyperparams.get("context_layers", 1),
                "final_layer": True,
            },
        })
        config["training_params"]["criterion"] = "MSE"
    elif model == "CNN":
        config.update({
            prefix + "model": "CNN",
            prefix + "model_params": {
                "in_channels": c_in,
                "out_channels": c_out,
                "out_conv_channels": hyperparams.get("out_conv_channels", 16 * 2 ** 4),
                "layers": hyperparams.get("layers", 4),
                "masking": hyperparams.get("masking", "none"),
                "dropout": hyperparams.get("dropout", 0),
                "embedding": hyperparams.get("embedding", "none"),
                "last_nonempty": hyperparams.get("last_nonempty", None)
            },
        })
    elif model == "CNN2":
        config.update({
            prefix + "model": "CNN2",
            prefix + "model_params": {
                "in_channels": c_in,
                "out_channels": c_out,
                "out_conv_channels": hyperparams.get("out_conv_channels", 16 * 2 ** 4),
                "layers": hyperparams.get("layers", 4),
                "masking": hyperparams.get("masking", "none"),
                "dropout": hyperparams.get("dropout", 0),
                "embedding": hyperparams.get("embedding", "none"),
                "last_nonempty": hyperparams.get("last_nonempty", None)
            },
        })
    elif model == "Iter":
        config.update({
            prefix + "model": "Iter",
            prefix + "model_params": {
                "batch_size": hyperparams.get("batch_size", 64),
                "seq_len": hyperparams.get("seq_len", 128),
                "in_channels": c_in
            },
        })
    else:
        raise ValueError("Model not supported")
