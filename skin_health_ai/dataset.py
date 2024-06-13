import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PH2Dataset(Dataset):
    def __init__(self, root_dir, indices=None, transform=None, size=(256, 256)):
        self.root_dir = root_dir
        self.transform = transform
        self.size = size
        self.images, self.lesions = self.load_images_and_lesions()
        if indices is not None:
            self.images = [self.images[i] for i in indices]
            self.lesions = [self.lesions[i] for i in indices]

    def load_images_and_lesions(self):
        images = []
        lesions = []
        for root, dirs, files in os.walk(os.path.join(self.root_dir, "PH2_Dataset")):
            if root.endswith("_Dermoscopic_Image"):
                images.append(imread(os.path.join(root, files[0])))
            if root.endswith("_lesion"):
                lesions.append(imread(os.path.join(root, files[0])))
        return images, lesions

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        lesion = self.lesions[idx]

        image = resize(image, self.size, mode="constant", anti_aliasing=True)
        lesion = resize(lesion, self.size, mode="constant", anti_aliasing=False) > 0.5

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        lesion = lesion[np.newaxis, ...].astype(np.float32)

        return torch.tensor(image), torch.tensor(lesion)


def get_dataloaders(root_dir, batch_size=8):
    dataset = PH2Dataset(root_dir=root_dir)
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

    data_tr = DataLoader(PH2Dataset(root_dir=root_dir, indices=train_indices), batch_size=batch_size, shuffle=True)
    data_val = DataLoader(PH2Dataset(root_dir=root_dir, indices=val_indices), batch_size=batch_size, shuffle=True)
    data_ts = DataLoader(PH2Dataset(root_dir=root_dir, indices=test_indices), batch_size=batch_size, shuffle=True)

    return data_tr, data_val, data_ts
