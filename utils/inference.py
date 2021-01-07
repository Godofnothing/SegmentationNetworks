import torch.nn.functional as F


def get_probabilities(y_pred):
    if len(y_pred.shape) == 3:
        return F.sigmoid(y_pred)
    else:
        return F.softmax(y_pred, dim=1)


def get_binary_outputs(y_prob, threshold = 0.5):
    return y_prob > threshold

