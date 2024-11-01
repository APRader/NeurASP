import time
import os
import pandas as pd
import torch

from torch.utils.data import Dataset
from os.path import join
from skimage import io
from torchvision import transforms
from network import Net
from neurasp import NeurASP

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

start_time = time.time()

class Follow_Suit(Dataset):

    def __init__(self, data_dir, examples, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.data = self.playing_cards = pd.read_csv(examples).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_idxs = self.data[index]
        imgs = []
        l = img_idxs[-1]
        for img_idx in img_idxs[:-1]:
            img_path = join(self.data_dir, f"{img_idx}.jpg")
            img = self.transform(io.imread(img_path))
            imgs.append(img)
        return imgs, l


transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((274, 174)),
            transforms.ToTensor(),
        ])

trainDataset = Follow_Suit(dir_path+'/../data/follow_suit/train', dir_path+'/../data/follow_suit/train.csv',
                           transform)

dataList = []
obsList = []
for [i1, i2, i3, i4], l in trainDataset:
    dataList.append({'i1': i1.unsqueeze(0), 'i2': i2.unsqueeze(0), 'i3': i3.unsqueeze(0), 'i4': i4.unsqueeze(0)})
    obsList.append(':- not winner({}).'.format(l))

#############################
# NeurASP program
#############################

dprogram = '''
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

% 4 Players
player(1..4).

% Definition of higher rank
rank_higher(P1, P2) :- card(P1, R1, _), card(P2, R2, _), rank_value(R1, V1), rank_value(R2, V2), V1 > V2.

% Link player's card to suit
suit(P1, S) :- card(P1, _, S).
'''

########
# Define nnMapping and optimizers, initialze NeurASP object
########

m = Net()
nnMapping = {'digit': m}
optimizers = {'digit': torch.optim.Adam(m.parameters(), lr=0.001)}
NeurASPobj = NeurASP(dprogram, nnMapping, optimizers)