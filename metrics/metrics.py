import torch


def iou(y_pred, y_true, eps=1e-7):
    y_pred = y_pred.squeeze(1).byte()
    y_true = y_true.squeeze(1).byte()

    intersection = (y_pred & y_true).float().sum()
    union = (y_pred | y_true).float().sum()

    return (intersection + eps) / (union + eps)