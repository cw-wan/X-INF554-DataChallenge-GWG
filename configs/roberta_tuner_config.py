import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Path:
    roberta = "modules/roberta-large"
    data = "data/"
    save = "checkpoints/roberta-tuner"


class Model:
    roberta_embsize = 1024
    output_size = 1


class DownStream:
    seed = 64
    batch_size = 64
    folds = 10
    learning_rate = 1e-6
    warm_up = 1
    total_epoch = 10
    decay = 1e-3
