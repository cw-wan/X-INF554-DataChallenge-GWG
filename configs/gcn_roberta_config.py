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
    rgcn_reg_basis = 10
    dropout = 0.5
    n_relations = 16
    gru = True
    gcn = False
    speaker_embedding = False
    bidirectional_gru = True
    n_speakers = 4


class DownStream:
    output_size = 1  # 1 for binary classification
    batch_size = 32
    dev_train_ratio = 0.1
    learning_rate = 1e-5
    warm_up = 1
    total_epoch = 20
    decay = 1e-4


class Seed:
    data_split = 64
