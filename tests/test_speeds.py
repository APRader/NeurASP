import unittest
import os
import torch
import time
import random

from neurasp import NeurASP
from unittest import mock
from slash import SLASH

from mvpp import MVPP
from mvpp_new import MVPP as MVPPNew
from mvpp_slash import MVPP as MVPPSlash

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
        from examples.mnistAdd.dataGen import dataList, obsList, train_dataset
        from examples.mnistAdd.network import Net

        dprogram = ("img(i1). img(i2).\n"
                    "addition(A,B,N) :- digit(0,A,N1), digit(0,B,N2), N=N1+N2.\n"
                    "nn(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).")

        slash_program = ("img(i1). img(i2).\n"
                         "addition(A,B,N):- digit(0,+A,-N1), digit(0,+B,-N2), N=N1+N2, A!=B.\n"
                         "npp(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).")

        m = Net()
        nnMapping = {'digit': m}
        optimizers = {'digit': torch.optim.Adam(m.parameters(), lr=0.001)}

        # Choose 1000 random examples
        idx_selection = random.sample(range(len(dataList)), 1000)
        dataList = [dataList[idx] for idx in idx_selection]
        obsList = [obsList[idx] for idx in idx_selection]

        # Original code
        NeurASPobj = NeurASP(dprogram, nnMapping, optimizers)
        with mock.patch.object(MVPP, 'prob_of_interpretation',
                               time_method(MVPP, 'prob_of_interpretation', 'prob_of_interpretation')):
            NeurASPobj.learn(dataList=dataList, obsList=obsList, epoch=1, smPickle=None, bar=True)

        NewrASPobj = NeurASP(dprogram, nnMapping, optimizers)

        # New code
        with (mock.patch('neurasp.MVPP', MVPPNew),
              mock.patch.object(MVPPNew, 'prob_of_interpretation',
                                time_method(MVPPNew, 'prob_of_interpretation', 'new_prob_of_interpretation'))):
            NewrASPobj.learn(dataList=dataList, obsList=obsList, epoch=1, smPickle=None, bar=True)

        # SLASH code
        dataList_slash = [{k: i.squeeze() for k, i in dataDict.items()} for dataDict in dataList]
        dataListLoader = torch.utils.data.DataLoader(list(zip(dataList_slash, obsList)))
        SLASHobj = SLASH(slash_program, nnMapping, optimizers, gpu = False)
        with mock.patch.object(MVPPSlash, 'prob_of_interpretation',
                               time_method(MVPPSlash, 'prob_of_interpretation', 'slash_prob_of_interpretation')):
            SLASHobj.learn(dataListLoader, 1)

        print(f"Old prob time: {sum(elapsed_times['prob_of_interpretation'])}")
        print(f"New prob time: {sum(elapsed_times['new_prob_of_interpretation'])}")
        print(f"SLASH prob time: {sum(elapsed_times['slash_prob_of_interpretation'])}")
