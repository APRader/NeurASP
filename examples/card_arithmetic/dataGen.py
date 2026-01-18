import os
import torch

import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from os.path import join
from torchvision import transforms, io

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)


class CardArithmetic(Dataset):

    def __init__(self, data_dir, downstream_labels, transform=None, latent_labels_file=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = downstream_labels

        if latent_labels_file:
            # Store latent labels if they are available
            latent_data = pd.read_csv(latent_labels_file)
            suits = np.array(['h', 'c', 's', 'd'])
            ranks = np.array(['2', '3', '4', '5', '6', '7', '8', '9', '10', 'j', 'q', 'k', 'a'])

            # Create an index to map semantic labels to numerical labels
            semantic_to_num = {}
            for s_idx, s in enumerate(suits):
                for r_idx, r in enumerate(ranks):
                    card_string = f"{r}{s}"
                    semantic_to_num[card_string] = r_idx + (s_idx * len(ranks))

            # Merge latent labels to downstream labels
            for i, player in enumerate(downstream_labels.columns[:-1], 1):
                self.data = pd.merge(self.data, latent_data, how='left', left_on=player, right_on='img')
                # Map semantic labels to numerical labels
                latent_col_name = f'latent_{i}'
                self.data[latent_col_name] = self.data['label'].map(semantic_to_num)
                # Drop redundant columns
                self.data = self.data.drop(columns=['img', 'label'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        imgs = []
        latent_labels = []
        img_idxs = self.data.iloc[index]
        for name, value in img_idxs.items():
            if name.startswith('player'):
                img_path = join(self.data_dir, f"{value}.jpg")
                img = self.transform(io.read_image(img_path))
                imgs.append(img)
            if name.startswith('latent'):
                latent_labels.append(value)
            if name == 'result':
                label = value
        if not latent_labels:
            return {'p': torch.stack(imgs)}, f':- not result({label}).'
        else:
            return {'p': (torch.stack(imgs), {'card': torch.Tensor(latent_labels)})}, f':- not result({label}).'


def split_dataset(data_file):
    data = pd.read_csv(data_file)
    val_data = data.sample(1000).reset_index(drop=True)
    val_idxs = set(pd.concat([val_data.iloc[:, 0], val_data.iloc[:, 1]]).unique())
    # As images might be in more than one row, we need to find all rows that include a val image
    val_rows = data.iloc[:, :-1].isin(val_idxs).any(axis=1)
    train_data = data[~val_rows].sample(10000).reset_index(drop=True)
    return train_data, val_data


def get_dataset(task_name):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((274, 174)),
        transforms.ToTensor(),
    ])

    train_data, val_data = split_dataset(dir_path + f'/data/{task_name}_labels.csv')

    trainDataset = CardArithmetic(dir_path + '/../../data/playing_cards/train',
                                  train_data, transform,
                                  dir_path + '/../../data/playing_cards/train/playing_card_labels_train.csv')

    valDataset = CardArithmetic(dir_path + '/../../data/playing_cards/train',
                                val_data, transform,
                                dir_path + '/../../data/playing_cards/train/playing_card_labels_train.csv')

    with open('data/playing_card_facts.lp') as file:
        facts = file.read()
    with open(f'data/{task_name}.lp') as file:
        task_rules = file.read()

    return trainDataset, valDataset, facts + '\n\n' + task_rules
