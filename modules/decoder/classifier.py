import torch.nn as nn


class BaseClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, name=None):
        super(BaseClassifier, self).__init__()
        self.name = name
        module_list = []
        for i, h in enumerate(hidden_size):
            if i == 0:
                module_list.append(nn.Linear(input_size, h))
                module_list.append(nn.GELU())
            else:
                module_list.append(nn.Linear(hidden_size[i - 1], h))
                module_list.append(nn.GELU())
        module_list.append(nn.Linear(hidden_size[-1], output_size))
        self.mlp = nn.Sequential(*module_list)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x
