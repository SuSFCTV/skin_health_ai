import torch


def bce_loss(y_real, y_pred):
    batch_size = y_real.shape[0]
    y_real = y_real.view(batch_size, -1)
    y_pred = y_pred.view(batch_size, -1)
    result = (y_pred - y_real * y_pred + torch.log(1 + torch.exp(-y_pred))).mean(-1)
    return result.mean()


def dice_loss(y_real, y_pred):
    smooth = 1e-8
    batch_size = y_pred.shape[0]
    y_pred = torch.sigmoid(y_pred)
    y_real = y_real.view(batch_size, -1)
    y_pred = y_pred.view(batch_size, -1)
    num = (2.0 * y_real * y_pred).sum(-1)
    den = (y_real + y_pred).sum(-1)
    res = 1 - ((num + smooth) / (den + smooth)).mean()
    return res


def focal_loss(y_real, y_pred, eps=1e-8, gamma=2):
    batch_size = y_real.shape[0]
    y_pred = torch.sigmoid(y_pred)
    y_pred = y_pred.view(batch_size, -1)
    y_real = y_real.view(batch_size, -1)
    result = (
        -((1 - y_pred) ** gamma * y_real * torch.log(y_pred) + y_pred**gamma * (1 - y_real) * torch.log(1 - y_pred))
    ).mean(-1)
    return result.mean()


def jaccard_loss(y_real, y_pred, smooth=1e-8):
    batch_size = y_pred.shape[0]
    y_pred = torch.sigmoid(y_pred)
    y_real = y_real.view(batch_size, -1)
    y_pred = y_pred.view(batch_size, -1)
    intersection = (y_real * y_pred).sum(-1)
    union = y_real.sum(-1) + y_pred.sum(-1) - intersection
    result = (intersection + smooth) / (union + smooth)
    return 1 - result.mean()
