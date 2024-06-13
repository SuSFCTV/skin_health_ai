import torch
from torch import optim
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, opt, loss_fn, metric, epochs, data_tr, data_val):
    model.to(device)
    losses = {"train": [], "val": []}
    metrics = {"train": [], "val": []}
    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
    for epoch in range(epochs):
        print("Epoch {}/{}:".format(epoch, epochs - 1), flush=True)
        for phase in ["train", "val"]:
            if phase == "train":
                dataloader = data_tr
                model.train()
            else:
                dataloader = data_val
                model.eval()
            running_loss = 0.0
            running_iou = 0.0
            for X_batch, Y_batch in tqdm(dataloader, leave=False, desc=f"{phase} iter:"):
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                opt.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    Y_pred = model(X_batch)
                    loss = loss_fn(Y_batch, Y_pred)
                    if phase == "train":
                        loss.backward()
                        opt.step()
                running_loss += loss.item()
                Y_pred = torch.sigmoid(Y_pred)
                Y_pred = (Y_pred > 0.5).float()
                running_iou += metric(Y_pred, Y_batch).mean().item()
            if phase == "train":
                lr_scheduler.step()
            epoch_loss = running_loss / len(dataloader)
            epoch_iou = running_iou / len(dataloader)
            losses[phase].append(epoch_loss)
            metrics[phase].append(epoch_iou)
            print("{} Loss: {:.4f}".format(phase, epoch_loss))
            print("{} IoU: {:.4f}".format(phase, epoch_iou))
    return losses, metrics
