import os

import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from skimage import io

class PlayingCards(Dataset):
    def __init__(self, data_dir, labels=None, labels_file=None, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        if labels_file:
            # Get data from file
            self.data = pd.read_csv(labels_file)
        elif labels is not None:
            # Get data from parameter
            self.data = labels
        else:
            raise Exception('No labels provided for playing card dataset')
        self.suits = np.array(['h', 'c', 's', 'd'])
        self.ranks = np.array(['2', '3', '4', '5', '6', '7', '8', '9', '10', 'j', 'q', 'k', 'a'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name = self.data['img'][index]
        img_path = os.path.join(self.data_dir, f"{img_name}.jpg")
        img = io.imread(img_path)
        if self.transform:
            img = self.transform(img)
        # Transform rank-suit labels into a numerical representation
        semantic_label = self.data['label'][index]
        number_label = 13 * np.where(self.suits == semantic_label[-1])[0].item() + \
                        np.where(self.ranks == semantic_label[:-1])[0].item()
        return img, number_label