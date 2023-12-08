import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Path:
    roberta = "modules/roberta-base"
    data = "data/"
    save = "checkpoints/"


class Model:
    bert_embsize = 1024
    utt_size = bert_embsize

    gru = False
    gru_layers = 2
    gru_bidirect = True

    temperature = 0.5
    dropout = 0.2

    gcn = True
    rgcn_reg_basis = 30
    graph_type = 1  # 1: discourse graph 2: DialogueGCN graph
    context_window = 8

    speaker_embedding = False
    speaker_embedding_size = 24
    n_speakers = 4
    if speaker_embedding:
        utt_size = utt_size + speaker_embedding_size

    speaker_relation = True

    shrink_before_gcn = True
    shrink_output_size = 300


class DownStream:
    custom_sampler = False if Model.graph_type == 2 else True  # Turn off custom sampler if using graph type 2
    seed = 64
    output_size = 1  # 1 for binary classification
    batch_size = 64
    dev_train_ratio = 0.2
    learning_rate = 1e-4
    warm_up = 1
    total_epoch = 10
    decay = 1e-3
