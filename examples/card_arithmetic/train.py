import argparse
import torch
import random
import os

import numpy as np

from newrasp import NeurASP
from examples.follow_suit.network import Net
from dataGen import get_dataset

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--learning_rate', '--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--checkpoint_freq', type=int, default=1000)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--data_dir', type=str, default="/data/public/playing_cards")
parser.add_argument('--output_dir', type=str, default="train_output")
parser.add_argument('--seed', type=int)
args = parser.parse_args()

if args.seed:
    seed = args.seed
else:
    # We generate a random number as the seed, so that the experiment run can still be reproduced
    seed = random.randint(0,100000)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Now that the seed is set, we can import the data
trainDataset, valDataset, dprogram = get_dataset(args.task, dir_path + args.data_dir)
trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=args.batch_size)
valLoader = torch.utils.data.DataLoader(valDataset, batch_size=args.batch_size)

m = Net()
nnMapping = {'card': m}
optimizers = {'card': torch.optim.Adam(m.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)}

NeurASPobj = NeurASP(dprogram, nnMapping, optimizers, gpu=True)
NeurASPobj.learn(trainLoader, epoch=args.epochs, lossFunc='semantic', accStep=args.checkpoint_freq,
                 bar=True, seed=seed, valDataset=valLoader, task=args.task)
# NeurASPobj.learn(trainDataset, epoch=args.epochs, storeSM=True, lossFunc='semantic', task='card_arithmetic',
#                  accStep=0, batchSize=args.batch_size, bar=True, seed=seed, valDataset=valDataset)