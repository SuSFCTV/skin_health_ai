import torch
from torch import optim

from skin_health_ai.config import EXTERNAL_DATA_DIR
from skin_health_ai.dataset import get_dataloaders
from skin_health_ai.metrics.iou import iou_pytorch
from skin_health_ai.models.train_utils.train import train
from skin_health_ai.models.train_utils.train_losses import bce_loss
from skin_health_ai.models.unet.unet_transposed import UNetTranspose


def main():
    data_tr, data_val, data_ts = get_dataloaders(root_dir=EXTERNAL_DATA_DIR / "PH2Dataset", batch_size=8)

    model = UNetTranspose()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.00001)
    loss_fn = bce_loss
    metric = iou_pytorch

    max_epochs = 20
    losses, metrics = train(model, optimizer, loss_fn, metric, max_epochs, data_tr, data_val)
    torch.save(model.state_dict(), "../../../models/unet_model.pth")


if __name__ == "__main__":
    main()
