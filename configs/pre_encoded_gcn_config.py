import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Path:
    roberta = "modules/roberta-base"
    data = "data/"
    save = "checkpoints/"


class Model:
    bert_embsize = 1024
    utt_size = bert_embsize

    temperature = 0.5
    dropout = 0.5

    gcn = True
    rgcn_reg_basis = 30

    n_speakers = 4
    # Switching between two styles of incorporating speaker information:
    # 1. Speaker Embedding
    speaker_embedding = True
    speaker_embedding_size = 1024
    # 2. Additional GCN layers for speaker turn relations
    speaker_relation = True

    # Whether to reduce the embedding dimension before feeding into GCN layers
    shrink_before_gcn = False
    shrink_output_size = 300


class DownStream:
    custom_sampler = True
    seed = 767576
    output_size = 1  # 1 for binary classification
    batch_size = 64
    dev_train_ratio = 0.2
    learning_rate = 1e-5
    warm_up = 1
    total_epoch = 8
    decay = 1e-3
