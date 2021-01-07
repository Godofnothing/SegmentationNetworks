def bce_loss(y_real, y_pred):
    # TODO
    # please don't use nn.BCELoss. write it from scratch
    return torch.sum(y_pred - y_pred * y_real + torch.log(1 + torch.exp(-y_pred)))