import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Path:
    roberta = "modules/roberta-base"
    data = "data/"
    save = "checkpoints/"


class Model:
    roberta_embedding_size = 768

    gru_hidden_size = 768
    gru_layers = 2

    n_speakers = 4
    n_relations = 16
    rgcn_reg_basis = 30

    dropout = 0.5

    gru = False
    gcn = True
    gat = False
    gat_heads = 4
    speaker_embedding = False
    bidirectional_gru = True


class DownStream:
    resample = False
    output_size = 1  # 1 for binary classification
    batch_size = 32
    dev_train_ratio = 0.1
    learning_rate = 1e-6
    warm_up = 1
    total_epoch = 20
    decay = 1e-3


class Seed:
    data_split = 64
