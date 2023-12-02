import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Path:
    roberta = "modules/roberta-base"
    data = "data/"
    save = "checkpoints/"


class Model:
    bert_embsize = 768
    dropout = 0.5
    gat_heads = 8
    edge_fea = 128
    ll_hidden_size = 64
    temperature = 0.5
    n_relations = 16


class DownStream:
    custom_sampler = True
    seed = 64
    output_size = 1  # 1 for binary classification
    batch_size = 8
    dev_train_ratio = 0.1
    learning_rate = 1e-6
    warm_up = 1
    total_epoch = 20
    decay = 1e-3
