import torch
import numpy as np
from torch.utils.data import DataLoader

from skin_health_ai.dataset import PH2Dataset
from skin_health_ai.models.unet.unet_transposed import load_model


def predict(model, data):
    with torch.no_grad():
        predictions = model(data)
    return predictions


def main():
    model = load_model("models/unet_model.pth")
    dataset = PH2Dataset(root_dir="data/external/PH2Dataset")
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    train_split = int(np.floor(0.6 * dataset_size))
    val_split = int(np.floor(0.2 * dataset_size))
    train_indices, val_indices, test_indices = (
        indices[:train_split],
        indices[train_split : train_split + val_split],
        indices[train_split + val_split :],
    )

    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    data_ts = DataLoader(dataset, batch_size=8, sampler=test_sampler)

    predictions = predict(model, data_ts)
    # Обработать предсказания


if __name__ == "__main__":
    main()
