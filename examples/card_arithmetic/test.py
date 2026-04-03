import argparse
import torch
import os
import pickle

import numpy as np
import pandas as pd

from newrasp import NeurASP
from examples.follow_suit.network import Net
from dataGen import get_dataset, CardArithmetic
from torchvision import transforms
from mvpp_new import MVPP

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str)
parser.add_argument('--seed', type=int)
args = parser.parse_args()

transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((274, 174)),
        transforms.ToTensor(),
    ])

data = pd.read_csv(f'{dir_path}/data/{args.task}_labels_test.csv')
testDataset = CardArithmetic(f'{dir_path}/../../data/playing_cards/test', data[:10000], transform,
                                  f'{dir_path}/../../data/playing_cards/test/playing_card_labels.csv')
dataLoader = torch.utils.data.DataLoader(testDataset, batch_size=64)

m = Net()
m.load_state_dict(torch.load(f'{dir_path}/saved_models/{args.task}_card_{args.seed}.pth', map_location=torch.device('cpu')))
nnMapping = {'card': m}
optimizers = {'card': torch.optim.Adam(m.parameters())}

with open(dir_path + '/data/playing_card_facts.lp') as file:
    facts = file.read()
with open(dir_path + f'/data/{args.task}.lp') as file:
    task_rules = file.read()
dprogram = facts + '\n' + task_rules
NeurASPobj = NeurASP(dprogram, nnMapping, optimizers, gpu=True)
dmvpp = MVPP(NeurASPobj.mvpp['program'])
with open(f'saved_models/{args.task}_stable_models.pkl', 'rb') as fp:
    NeurASPobj.stableModels = pickle.load(fp)
downAcc, latentAcc = NeurASPobj.calculate_accuracies(dataLoader, dmvpp)
print(f"Downstream acc: {downAcc}")
print(f"Latent acc: {latentAcc}")
