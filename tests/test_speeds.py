import unittest
import os
import torch
import time
import random

import numpy as np

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

    def test_speed_synthetic(self, num_models=1000, num_inputs=20, num_concepts=40):
        models_new = np.random.randint(0, num_concepts, (num_models, num_inputs))
        models = [[f"test(i{idx},{value})" for idx, value in enumerate(model)] for model in models_new]
        model_idx_list = [[(idx, value) for idx, value in enumerate(model)] for model in models_new]
        pc = [[f"test(i{image},{value})" for value in range(num_concepts)] for image in range(num_inputs)]
        parameters = np.random.random_sample((num_inputs, num_concepts)).tolist()
        selection_mask = torch.tensor([[True for _ in range(num_concepts)] for _ in range(num_inputs)])

        mock_return = (pc, parameters, False, "mock_asp", "mock_pi", "mock_remain_probs")

        with (mock.patch.object(MVPP, 'parse', return_value=mock_return),
              mock.patch.object(MVPP, 'normalize_probs')):
            mvpp = MVPP('')
            start_time = time.perf_counter()
            probs = [mvpp.prob_of_interpretation(model) for model in models]
            neurasp_prob_time = time.perf_counter() - start_time
            [mvpp.mvppLearnRule(ruleIdx, models, probs) for ruleIdx in range(num_inputs)]
            neurasp_grad_time = time.perf_counter() - neurasp_prob_time - start_time

        with mock.patch.object(MVPPSlash, 'parse', return_value=mock_return + ([],)):
            mvpp_slash = MVPPSlash('')
            mvpp_slash.max_n = num_concepts
            mvpp_slash.M = torch.tensor(parameters)
            mvpp_slash.binary_rule_belongings = {}
            mvpp_slash.selection_mask = selection_mask

            start_time = time.perf_counter()
            probs = []
            for i in range(len(models)):
                probs.append(mvpp_slash.prob_of_interpretation(models[i], model_idx_list[i]))
            slash_prob_time = time.perf_counter() - start_time
            mvpp_slash.mvppLearnRule(models, model_idx_list, 'cpu', torch.tensor(probs))
            slash_grad_time = time.perf_counter() - slash_prob_time - start_time

        with (mock.patch.object(MVPPNew, 'parse', return_value=mock_return),
              mock.patch.object(MVPPNew, 'normalize_probs')):
            mvpp_new = MVPPNew('')
            start_time = time.perf_counter()
            probs = mvpp_new.prob_of_interpretation(models_new)
            newrasp_prob_time = time.perf_counter() - start_time
            mvpp_new.mvppLearnRule(models_new, np.array(probs), 5)
            newrasp_grad_time = time.perf_counter() - newrasp_prob_time - start_time

        print(f"Old prob time: {neurasp_prob_time}")
        print(f"SLASH prob time: {slash_prob_time}")
        print(f"New prob time: {newrasp_prob_time}")
        print("==========")
        print(f"Old grad time: {neurasp_grad_time}")
        print(f"SLASH grad time: {slash_grad_time}")
        print(f"New grad time: {newrasp_grad_time}")
        print("\n")

        assert (newrasp_prob_time + newrasp_grad_time < neurasp_prob_time + neurasp_grad_time)
        assert (newrasp_prob_time + newrasp_grad_time < slash_prob_time + slash_grad_time)

    def test_speeds_synthetic(self):
        """Test speeds of different implementations of the probability calculations with synthetic data"""

        # 100 models with 10 inputs and 15 possible concepts
        print("100 models:")
        self.test_speed_synthetic(100, 10, 15)

        # 1000 models with 18 inputs and 27 possible concepts
        print("1,000 models:")
        self.test_speed_synthetic(1000, 18, 27)

        # 10000 models with 25 inputs and 40 possible concepts
        print("10,000 models:")
        self.test_speed_synthetic(10000, 25, 40)

    def test_speeds_mnist_add(self):
        """Test speeds of different implementations for the MNIST Addition task"""
        os.chdir('../examples/mnistAdd')
        from examples.mnistAdd.dataGen import dataList, obsList
        from examples.mnistAdd.network import Net

        dprogram = ("img(i1). img(i2).\n"
                    "addition(A,B,N) :- digit(0,A,N1), digit(0,B,N2), N=N1+N2.\n"
                    "nn(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).")

        slash_program = ("img(i1). img(i2).\n"
                         "addition(A,B,N):- digit(0,+A,-N1), digit(0,+B,-N2), N=N1+N2, A!=B.\n"
                         "npp(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).")

        m = Net()
        nnMapping = {'digit': m}
        optimizers = {'digit': torch.optim.Adam(m.parameters())}

        # Choose 1000 random examples
        idx_selection = random.sample(range(len(dataList)), 1000)
        dataList = [dataList[idx] for idx in idx_selection]
        obsList = [obsList[idx] for idx in idx_selection]

        # Original code
        NeurASPobj = NeurASP(dprogram, nnMapping, optimizers)
        with (mock.patch.object(MVPP, 'prob_of_interpretation',
                                time_method(MVPP, 'prob_of_interpretation', 'mnist_add_prob')),
              mock.patch.object(MVPP, 'mvppLearnRule',
                                time_method(MVPP, 'mvppLearnRule', 'mnist_add_grad'))):
            start_time = time.perf_counter()
            NeurASPobj.learn(dataList=dataList, obsList=obsList, epoch=1)
            neurasp_time = time.perf_counter() - start_time

        # SLASH code
        dataList_slash = [{k: i.squeeze() for k, i in dataDict.items()} for dataDict in dataList]
        dataListLoader = torch.utils.data.DataLoader(list(zip(dataList_slash, obsList)))
        SLASHobj = SLASH(slash_program, nnMapping, optimizers, gpu=False)
        with (mock.patch.object(MVPPSlash, 'prob_of_interpretation',
                                time_method(MVPPSlash, 'prob_of_interpretation', 'slash_mnist_add_prob')),
              mock.patch.object(MVPPSlash, 'mvppLearnRule',
                                time_method(MVPPSlash, 'mvppLearnRule', 'slash_mnist_add_grad'))):
            start_time = time.perf_counter()
            SLASHobj.learn(dataListLoader, 1)
            slash_time = time.perf_counter() - start_time

        # New code
        NewrASPobj = NeurASP(dprogram, nnMapping, optimizers)
        with (mock.patch('neurasp.MVPP', MVPPNew),
              mock.patch.object(MVPPNew, 'prob_of_interpretation',
                                time_method(MVPPNew, 'prob_of_interpretation', 'new_mnist_add_prob')),
              mock.patch.object(MVPPNew, 'mvppLearnRule',
                                time_method(MVPPNew, 'mvppLearnRule', 'new_mnist_add_grad'))):
            start_time = time.perf_counter()
            NewrASPobj.learn(dataList=dataList, obsList=obsList, epoch=1)
            newrasp_time = time.perf_counter() - start_time

        print("\n")
        print(f"Old prob time: {sum(elapsed_times['mnist_add_prob'])}")
        print(f"SLASH prob time: {sum(elapsed_times['slash_mnist_add_prob'])}")
        print(f"New prob time: {sum(elapsed_times['new_mnist_add_prob'])}")
        print("==========")
        print(f"Old grad time: {sum(elapsed_times['mnist_add_grad'])}")
        print(f"SLASH grad time: {sum(elapsed_times['slash_mnist_add_grad'])}")
        print(f"New grad time: {sum(elapsed_times['new_mnist_add_grad'])}")
        print("==========")
        print(f"Total NeurASP time: {neurasp_time}")
        print(f"Total SLASH time: {slash_time}")
        print(f"Total new time: {newrasp_time}")

        # New code should be faster than existing code
        assert (newrasp_time < neurasp_time)
        assert (newrasp_time < slash_time)

    def test_speeds_top_k(self):
        """Test speeds of different implementations for the Top Knapsack task"""
        os.chdir('../examples/top_k')
        from examples.top_k.dataGen import dataList, obsList
        from examples.top_k.network import FC

        dprogram = ("nn(in(10, k), [true, false]).\n"
                    "% define maxweight k\n"
                    "#const k = 7.\n"
                    ":- #sum{1, I : in(I,k,true)} > k.")

        m = FC(10, 50, 50, 50, 50, 50, 10)
        nnMapping = {'in': m}
        optimizers = {'in': torch.optim.Adam(m.parameters(), lr=0.001)}

        # Choose 1000 random examples
        idx_selection = random.sample(range(len(dataList)), 1000)
        dataList = [dataList[idx] for idx in idx_selection]
        obsList = [obsList[idx] for idx in idx_selection]

        # Original code
        NeurASPobj = NeurASP(dprogram, nnMapping, optimizers)
        with (mock.patch.object(MVPP, 'prob_of_interpretation',
                                time_method(MVPP, 'prob_of_interpretation', 'top_k_prob')),
              mock.patch.object(MVPP, 'mvppLearnRule',
                                time_method(MVPP, 'mvppLearnRule', 'top_k_grad'))):
            start_time = time.perf_counter()
            NeurASPobj.learn(dataList=dataList, obsList=obsList, epoch=1, smPickle=None)
            neurasp_time = time.perf_counter() - start_time

        # New code
        NewrASPobj = NeurASP(dprogram, nnMapping, optimizers)
        with (mock.patch('neurasp.MVPP', MVPPNew),
              mock.patch.object(MVPPNew, 'prob_of_interpretation',
                                time_method(MVPPNew, 'prob_of_interpretation', 'new_top_k_prob')),
              mock.patch.object(MVPPNew, 'mvppLearnRule',
                                time_method(MVPPNew, 'mvppLearnRule', 'new_top_k_grad'))):
            start_time = time.perf_counter()
            NewrASPobj.learn(dataList=dataList, obsList=obsList, epoch=1)
            newrasp_time = time.perf_counter() - start_time

        print("\n")
        print(f"Old prob time: {sum(elapsed_times['top_k_prob'])}")
        print(f"New prob time: {sum(elapsed_times['new_top_k_prob'])}")
        print("==========")
        print(f"Old grad time: {sum(elapsed_times['top_k_grad'])}")
        print(f"New grad time: {sum(elapsed_times['new_top_k_grad'])}")
        print("==========")
        print(f"Total NeurASP time: {neurasp_time}")
        print(f"Total new time: {newrasp_time}")

        # New code should be faster than existing code
        assert (newrasp_time < neurasp_time)

    def test_add2x2(self):
        """Test speeds of different implementations for the Add 2x2 task"""
        os.chdir('../examples/add2x2')
        from examples.add2x2.dataGen import dataList, obsList
        from examples.add2x2.network import Net

        dprogram = ("nn(digit(4,i), [0,1,2,3,4,5,6,7,8,9]).\n"
                    "add2x2(R1,R2,C1,C2) :- digit(0,i,N1), digit(1,i,N2), digit(2,i,N3), digit(3,i,N4), "
                    "R1=N1+N2, R2=N3+N4, C1=N1+N3, C2=N2+N4.")

        m = Net()
        nnMapping = {'digit': m}
        optimizers = {'digit': torch.optim.Adam(m.parameters())}

        # Choose 1000 random examples
        idx_selection = random.sample(range(len(dataList)), 1000)
        dataList = [dataList[idx] for idx in idx_selection]
        obsList = [obsList[idx] for idx in idx_selection]

        # Original code
        NeurASPobj = NeurASP(dprogram, nnMapping, optimizers)
        with (mock.patch.object(MVPP, 'prob_of_interpretation',
                                time_method(MVPP, 'prob_of_interpretation', 'add2x2_prob')),
              mock.patch.object(MVPP, 'mvppLearnRule',
                                time_method(MVPP, 'mvppLearnRule', 'add2x2_grad'))):
            start_time = time.perf_counter()
            NeurASPobj.learn(dataList=dataList, obsList=obsList, epoch=1, smPickle=None)
            neurasp_time = time.perf_counter() - start_time

        # New code
        NewrASPobj = NeurASP(dprogram, nnMapping, optimizers)
        with (mock.patch('neurasp.MVPP', MVPPNew),
              mock.patch.object(MVPPNew, 'prob_of_interpretation',
                                time_method(MVPPNew, 'prob_of_interpretation', 'new_add2x2_prob')),
              mock.patch.object(MVPPNew, 'mvppLearnRule',
                                time_method(MVPPNew, 'mvppLearnRule', 'new_add2x2_grad'))):
            start_time = time.perf_counter()
            NewrASPobj.learn(dataList=dataList, obsList=obsList, epoch=1)
            newrasp_time = time.perf_counter() - start_time

        print("\n")
        print(f"Old prob time: {sum(elapsed_times['add2x2_prob'])}")
        print(f"New prob time: {sum(elapsed_times['new_add2x2_prob'])}")
        print("==========")
        print(f"Old grad time: {sum(elapsed_times['add2x2_grad'])}")
        print(f"New grad time: {sum(elapsed_times['new_add2x2_grad'])}")
        print("==========")
        print(f"Total NeurASP time: {neurasp_time}")
        print(f"Total new time: {newrasp_time}")

        # New code should be faster than existing code
        assert (newrasp_time < neurasp_time)

    def test_speeds_follow_suit(self):
        """Test speeds of different implementations for Follow Suit task.
        WARNING: This test takes hours to complete!"""
        os.chdir('../examples/follow_suit')
        from examples.follow_suit.dataGen import dataList, obsList, facts, rules, dprogram
        from examples.follow_suit.network import Net

        m = Net()
        nnMapping = {'card': m}
        optimizers = {'card': torch.optim.Adam(m.parameters())}

        # Choose 1 random example
        idx_selection = random.sample(range(len(dataList)), 1)
        dataList = [dataList[idx] for idx in idx_selection]
        obsList = [obsList[idx] for idx in idx_selection]

        # Original code
        NeurASPobj = NeurASP(dprogram, nnMapping, optimizers)
        with (mock.patch.object(MVPP, 'prob_of_interpretation',
                                time_method(MVPP, 'prob_of_interpretation', 'follow_suit_prob')),
              mock.patch.object(MVPP, 'mvppLearnRule',
                                time_method(MVPP, 'mvppLearnRule', 'follow_suit_grad'))):
            start_time = time.perf_counter()
            NeurASPobj.learn(dataList=dataList, obsList=obsList, epoch=1)
            neurasp_time = time.perf_counter() - start_time

        # SLASH code
        dataList_slash = [{k: i.squeeze() for k, i in dataDict.items()} for dataDict in dataList]
        dataListLoader = torch.utils.data.DataLoader(list(zip(dataList_slash, obsList)))
        slash_neural_preds = (
            "\nnpp(card(1,P), [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,"
            "27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51]) :- player(P)."
            "\nsuit(P,h) :- card(0,+P,-C), C <= 12."
            "\nsuit(P,c) :- card(0,+P,-C), C >= 13, C <= 25."
            "\nsuit(P,s) :- card(0,+P,-C), C >= 26, C <= 38."
            "\nsuit(P,d) :- card(0,+P,-C), C >= 39."
            "\nrank(P,R) :- card(0,+P,-C), rank_value(R,-C\\13+2).")
        slash_program = facts + rules + slash_neural_preds
        SLASHobj = SLASH(slash_program, nnMapping, optimizers, gpu=False)
        with (mock.patch.object(MVPPSlash, 'prob_of_interpretation',
                                time_method(MVPPSlash, 'prob_of_interpretation', 'slash_follow_suit_prob')),
              mock.patch.object(MVPPSlash, 'mvppLearnRule',
                                time_method(MVPPSlash, 'mvppLearnRule', 'slash_follow_suit_grad'))):
            start_time = time.perf_counter()
            SLASHobj.learn(dataListLoader, 1)
            slash_time = time.perf_counter() - start_time

        # New code
        NewrASPobj = NeurASP(dprogram, nnMapping, optimizers)
        with (mock.patch('neurasp.MVPP', MVPPNew),
              mock.patch.object(MVPPNew, 'prob_of_interpretation',
                                time_method(MVPPNew, 'prob_of_interpretation', 'new_follow_suit_prob')),
              mock.patch.object(MVPPNew, 'mvppLearnRule',
                                time_method(MVPPNew, 'mvppLearnRule', 'new_follow_suit_grad'))):
            start_time = time.perf_counter()
            NewrASPobj.learn(dataList=dataList, obsList=obsList, epoch=1)
            newrasp_time = time.perf_counter() - start_time

        print("\n")
        print(f"Old prob time: {sum(elapsed_times['follow_suit_prob'])}")
        print(f"SLASH prob time: {sum(elapsed_times['slash_follow_suit_prob'])}")
        print(f"New prob time: {sum(elapsed_times['new_follow_suit_prob'])}")
        print("==========")
        print(f"Old grad time: {sum(elapsed_times['follow_suit_grad'])}")
        print(f"SLASH grad time: {sum(elapsed_times['slash_follow_suit_grad'])}")
        print(f"New grad time: {sum(elapsed_times['new_follow_suit_grad'])}")
        print("==========")
        print(f"Total NeurASP time: {neurasp_time}")
        print(f"Total SLASH time: {slash_time}")
        print(f"Total new time: {newrasp_time}")

        # New code should be faster than existing code
        assert (newrasp_time < neurasp_time)
        assert (newrasp_time < slash_time)
