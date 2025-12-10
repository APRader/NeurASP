import os
import torch
import argparse

import pandas as pd

from torch.utils.data import Dataset
from os.path import join
from torchvision import transforms, io
from newrasp import NeurASP
from data.playing_cards.dataset import PlayingCards
from examples.follow_suit.network import Net

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

class CardArithmetic(Dataset):

    def __init__(self, data_dir, labels_file, transform=None, latent_labels_file=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = pd.read_csv(labels_file)

        if latent_labels_file:
            # Store latent labels if they are available
            latent_data = pd.read_csv(latent_labels_file)

            # Create train set out of all the images that are in the downstream dataset
            train_idxs = set(pd.concat([self.data.iloc[:, 0], self.data.iloc[:, 1]]).unique())
            is_in_train_mask = latent_data['img'].isin(train_idxs)
            latent_train_labels = latent_data[is_in_train_mask].copy().reset_index(drop=True)

            # The validation set consists of images not in the downstream dataset, and is capped at 1000 entries
            latent_val_labels = latent_data[~is_in_train_mask].copy()
            latent_val_labels = latent_val_labels.sample(n=min(len(latent_val_labels), 1000)).reset_index(drop=True)

            self.latent_train_data = {'card': PlayingCards(data_dir, labels=latent_train_labels, transform=transform)}
            self.latent_val_data = {'card': PlayingCards(data_dir, labels=latent_val_labels, transform=transform)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        imgs = []
        img_idxs = self.data.iloc[index]
        l = img_idxs.iloc[-1]
        for img_idx in img_idxs[:-1]:
            img_path = join(self.data_dir, f"{img_idx}.jpg")
            img = self.transform(io.read_image(img_path))
            imgs.append(img)
        return {'p': torch.stack(imgs)}, f':- not result({l}).'


transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((274, 174)),
            transforms.ToTensor(),
        ])

trainDataset = CardArithmetic(dir_path + '/../../data/playing_cards/train',
                              dir_path +'/data/card_arithmetic_2p_image_labels_30k.csv', transform,
                              dir_path +'/../../data/playing_cards/train/playing_card_labels_train.csv')

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

suit_value(d,1). suit_value(c,2). suit_value(s,3). suit_value(h,4).

% 2 Players
% player(p1). player(p2).'''

rules = '''
result(X) :- suit(P1,S1), suit(P2,S2), suit_value(S1,SV1), suit_value(S2, SV2), 
             rank(P1,R1), rank(P2,R2), rank_value(R1,V1), rank_value(R2,V2), 
             X = V1*SV1 + V2*SV2, P1!=P2.'''

neural_preds = '''
nn(card(2,p), [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51]).
suit(P,h) :- card(P,p,C), C <= 12.
suit(P,c) :- card(P,p,C), C >= 13, C <= 25.
suit(P,s) :- card(P,p,C), C >= 26, C <= 38.
suit(P,d) :- card(P,p,C), C >= 39.
rank(P,R) :- card(P,p,C), rank_value(R,C\\13+2).'''

dprogram = facts + rules + neural_preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--checkpoint_freq', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default="train_output")
    args = parser.parse_args()

    m = Net()
    nnMapping = {'card': m}
    optimizers = {'card': torch.optim.Adam(m.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)}

    NeurASPobj = NeurASP(dprogram, nnMapping, optimizers, gpu=True)
    NeurASPobj.learn(trainDataset, epoch=5, smPickle='card_arithmetic_2p_stable_models.pkl',
                     accStep=args.checkpoint_freq, batchSize=args.batch_size)