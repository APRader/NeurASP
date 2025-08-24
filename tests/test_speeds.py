import unittest
import os
import torch
import time
import random

from neurasp import NeurASP
from unittest import mock
from mvpp import MVPP

elapsed_times = {}

def time_method(class_obj, method_name, elapsed_times_key):
    original = getattr(class_obj, method_name)

    # Create entry for this method in elapsed times
    elapsed_times[elapsed_times_key] = []

    def wrapper(self, *args, **kwargs):
        start_time = time.perf_counter()
        result = original(self, *args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        elapsed_times[elapsed_times_key].append(elapsed_time)
        return result

    return wrapper

class TestSpeeds(unittest.TestCase):

    def test_mnist_add(self):
        """Test speed for MNIST Addition task"""
        os.chdir('examples/mnistAdd')
        from examples.mnistAdd.dataGen import dataList, obsList
        from examples.mnistAdd.network import Net

        dprogram = ("img(i1). img(i2).\n"
                    "addition(A,B,N) :- digit(0,A,N1), digit(0,B,N2), N=N1+N2.\n"
                    "nn(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).")

        m = Net()
        nnMapping = {'digit': m}
        optimizers = {'digit': torch.optim.Adam(m.parameters(), lr=0.001)}
        NeurASPobj = NeurASP(dprogram, nnMapping, optimizers)

        # Choose 100 random examples
        idx_selection = random.sample(range(len(dataList)), 1000)
        dataList = [dataList[idx] for idx in idx_selection]
        obsList = [obsList[idx] for idx in idx_selection]
        with mock.patch.object(MVPP, 'prob_of_interpretation', new=time_method(MVPP, 'prob_of_interpretation', 'mvpp_prob_of_interpretation')):
            NeurASPobj.learn(dataList=dataList, obsList=obsList, epoch=1, smPickle=None, bar=True)

        print(f"Time to calculate 1000 probs of interpretations: {sum(elapsed_times['mvpp_prob_of_interpretation'])}" )