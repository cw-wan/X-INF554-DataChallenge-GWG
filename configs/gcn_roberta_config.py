import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Path:
    roberta = "modules/roberta-base"
    data = "data/"
    save = "checkpoints/"


class Model:
    bert_embsize = 768

    gru = False
    gru_layers = 2
    gru_bidirect = True

    temperature = 0.5
    dropout = 0.5

    gcn = True
    rgcn_reg_basis = 30
    graph_type = 1  # 1: discourse graph 2: DialogueGCN graph
    context_window = 8

    speaker_embedding = False
    n_speakers = 4


class DownStream:
    custom_sampler = False if Model.graph_type == 2 else True  # Turn off custom sampler if using graph type 2
    seed = 64
    output_size = 1  # 1 for binary classification
    batch_size = 64
    dev_train_ratio = 0.1
    learning_rate = 1e-6
    warm_up = 1
    total_epoch = 20
    decay = 1e-3
