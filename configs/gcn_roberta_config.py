import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Path:
    roberta = "modules/roberta-base"
    data = "data/"
    save = "checkpoints/"


class Model:
    bert_embsize = 768

    gru_layers = 2
    gru_bidirect = True

    n_speakers = 4
    n_relations = 16
    rgcn_reg_basis = 30

    temperature = 0.5

    dropout = 0.5

    gru = False
    gcn = True
    speaker_embedding = False


class DownStream:
    seed = 64
    output_size = 1  # 1 for binary classification
    batch_size = 32
    dev_train_ratio = 0.1
    learning_rate = 1e-6
    warm_up = 1
    total_epoch = 20
    decay = 1e-3


class Seed:
    data_split = 64
