import torch


def f1_loss(y_pred, y_true):
    y_pred = torch.mul(torch.sub(y_pred, torch.ones_like(y_pred), alpha=0.5), 2)
    y_pred = torch.pow(y_pred, 3)
    y_pred = torch.add(torch.mul(y_pred, 0.5), torch.ones_like(y_pred), alpha=0.5)

    tp = torch.sum(y_pred * y_true)
    tn = torch.sum((1 - y_pred) * (1 - y_true))
    fp = torch.sum(y_pred * (1 - y_true))
    fn = torch.sum((1 - y_pred) * y_true)

    p = tp / (tp + fp + 1e-10)
    r = tp / (tp + fn + 1e-10)

    f1 = 2 * p * r / (p + r + 1e-10)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    return 1 - torch.mean(f1)
