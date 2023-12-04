import torch
import torch.nn as nn


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


def diff_loss(input1, input2):
    # Adapted from https://github.com/BladeDancer957/DualGATs/blob/main/model_utils.py
    # input1 (B,D)    input2 (B,D)

    # Zero mean
    input1_mean = torch.mean(input1, dim=0, keepdims=True)  # (1,D)
    input2_mean = torch.mean(input2, dim=0, keepdims=True)  # (1,D)
    input1 = input1 - input1_mean  # (B,D)
    input2 = input2 - input2_mean  # (B,D)

    input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()  # (B,1)
    input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)  # (B,D)

    input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()  # (B,1)
    input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)  # (B,D)

    loss = 1.0 / (torch.mean(torch.norm(input1_l2 - input2_l2, p=2, dim=1)))

    return loss
