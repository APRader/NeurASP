import os
import torch

import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from os.path import join
from torchvision import transforms, io
from data.playing_cards.dataset import PlayingCards

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


transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((274, 174)),
            transforms.ToTensor(),
        ])

train_data, val_data = split_dataset(dir_path + '/data/card_arithmetic_2p_image_labels_30k.csv')

trainDataset = CardArithmetic(dir_path + '/../../data/playing_cards/train',
                              train_data, transform,
                              dir_path + '/../../data/playing_cards/train/playing_card_labels_train.csv')

valDataset = CardArithmetic(dir_path + '/../../data/playing_cards/train',
                            val_data, transform,
                            dir_path + '/../../data/playing_cards/train/playing_card_labels_train.csv')

#############################
# NeurASP program
#############################

facts = '''
% Suits
suit(h).
suit(s).
suit(d).
suit(c).

% Ranks
rank(a).
rank(2).
rank(3).
rank(4).
rank(5).
rank(6).
rank(7).
rank(8).
rank(9).
rank(10).
rank(j).
rank(q).
rank(k).

% Rank Value
rank_value(2, 2).
rank_value(3, 3).
rank_value(4, 4).
rank_value(5, 5).
rank_value(6, 6).
rank_value(7, 7).
rank_value(8, 8).
rank_value(9, 9).
rank_value(10, 10).
rank_value(j, 11).
rank_value(q, 12).
rank_value(k, 13).
rank_value(a, 14).

% suit_value(d,0). suit_value(c,13). suit_value(s,26). suit_value(h,39).
suit_value(d,1). suit_value(c,2). suit_value(s,3). suit_value(h,4).

% 2 Players
% player(p1). player(p2).'''

rules = '''
result(X) :- suit(P1,S1), suit(P2,S2), suit_value(S1,SV1), suit_value(S2, SV2), 
             rank(P1,R1), rank(P2,R2), rank_value(R1,V1), rank_value(R2,V2), 
%             X = V1 + SV1 + V2 + SV2, P1!=P2.
             X = V1 * SV1 + V2 * SV2, P1!=P2.'''

neural_preds = '''
nn(card(2,p), [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51]).
suit(P,h) :- card(P,p,C), C <= 12.
suit(P,c) :- card(P,p,C), C >= 13, C <= 25.
suit(P,s) :- card(P,p,C), C >= 26, C <= 38.
suit(P,d) :- card(P,p,C), C >= 39.
rank(P,R) :- card(P,p,C), rank_value(R,C\\13+2).'''

dprogram = facts + rules + neural_preds
