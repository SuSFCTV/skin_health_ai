import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score_model(model, metric, data):
    model.to(device)
    model.eval()
    scores = 0
    for X_batch, Y_batch in data:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        Y_pred = model(X_batch)
        Y_pred = torch.sigmoid(Y_pred)
        Y_pred = (Y_pred > 0.5).float()
        scores += metric(Y_pred, Y_batch).mean().item()
    return scores / len(data)
