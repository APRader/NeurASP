import argparse
import torch
import random

from newrasp import NeurASP
from examples.follow_suit.network import Net

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--checkpoint_freq', type=int, default=1000)
parser.add_argument('--output_dir', type=str, default="train_output")
parser.add_argument('--seed', type=int)
args = parser.parse_args()

if args.seed:
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
else:
    # We generate a random number as the seed, so that the experiment run can still be reproduced
    seed = random.randint(0,100000)
    torch.manual_seed(seed)
    random.seed(seed)

# Now that the seed is set, we can import the data
from dataGen import trainDataset, valDataset, dprogram

m = Net()
nnMapping = {'card': m}
optimizers = {'card': torch.optim.Adam(m.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)}

NeurASPobj = NeurASP(dprogram, nnMapping, optimizers, gpu=True)
NeurASPobj.learn(trainDataset, epoch=5, smPickle='card_arithmetic_2p_stable_models.pkl',
                 accStep=args.checkpoint_freq, batchSize=args.batch_size, bar=True, seed=seed, valDataset=valDataset)