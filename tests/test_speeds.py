import unittest
import os
import torch
import time
import random

import numpy as np

from neurasp import NeurASP
from newrasp import NeurASP as NewrASP
from unittest import mock
from slash import SLASH

from mvpp import MVPP
from mvpp_new import MVPP as MVPPNew
from mvpp_slash import MVPP as MVPPSlash

elapsed_times = {}
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


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


def sample_examples(dataList, obsList, sample_size):
    # Sample random subset of examples
    idx_selection = random.sample(range(len(dataList)), sample_size)
    dataList = [dataList[idx] for idx in idx_selection]
    obsList = [obsList[idx] for idx in idx_selection]
    return dataList, obsList


def measure_neurasp_speed(dprogram, nnMapping, optimizers, dataList, obsList, example_name, opt=False):
    """Measure the speed of the original NeurASP code for an example."""
    NeurASPobj = NeurASP(dprogram, nnMapping, optimizers)
    with (mock.patch.object(MVPP, 'find_k_SM_under_obs',
                            time_method(MVPP, 'find_k_SM_under_obs', f'{example_name}_model')),
          mock.patch.object(MVPP, 'find_all_opt_SM_under_obs_WC',
                            time_method(MVPP, 'find_all_opt_SM_under_obs_WC', f'{example_name}_model')),
          mock.patch.object(MVPP, 'prob_of_interpretation',
                            time_method(MVPP, 'prob_of_interpretation', f'{example_name}_prob')),
          mock.patch.object(MVPP, 'mvppLearnRule',
                            time_method(MVPP, 'mvppLearnRule', f'{example_name}_grad'))):
        start_time = time.perf_counter()
        NeurASPobj.learn(dataList=dataList, obsList=obsList, epoch=1, opt=opt)
        return time.perf_counter() - start_time


def measure_slash_speed(dprogram, nnMapping, optimizers, dataListLoader, example_name):
    """Measure the speed of the SLASH code for an example."""
    SLASHobj = SLASH(dprogram, nnMapping, optimizers, gpu=False)
    with (mock.patch.object(MVPPSlash, 'find_k_SM_under_query',
                            time_method(MVPPSlash, 'find_k_SM_under_query', f'slash_{example_name}_model')),
          mock.patch.object(MVPPSlash, 'prob_of_interpretation',
                            time_method(MVPPSlash, 'prob_of_interpretation', f'slash_{example_name}_prob')),
          mock.patch.object(MVPPSlash, 'mvppLearnRule',
                            time_method(MVPPSlash, 'mvppLearnRule', f'slash_{example_name}_grad'))):
        start_time = time.perf_counter()
        SLASHobj.learn(dataListLoader, 1)
        return time.perf_counter() - start_time


def measure_newrasp_speed(dprogram, nnMapping, optimizers, dataList, obsList, example_name, opt=False):
    """Measure the speed of the new implementation of NeurASP for an example."""
    NewrASPobj = NewrASP(dprogram, nnMapping, optimizers)
    with (mock.patch.object(MVPPNew, 'find_k_SM_under_obs',
                            time_method(MVPPNew, 'find_k_SM_under_obs', f'new_{example_name}_model')),
          mock.patch.object(MVPPNew, 'prob_of_interpretation',
                            time_method(MVPPNew, 'prob_of_interpretation', f'new_{example_name}_prob')),
          mock.patch.object(MVPPNew, 'mvppLearnRule',
                            time_method(MVPPNew, 'mvppLearnRule', f'new_{example_name}_grad'))):
        dataset = list(zip(dataList, obsList))
        start_time = time.perf_counter()
        NewrASPobj.learn(dataset, epoch=1, opt=opt)
        return time.perf_counter() - start_time


def print_times(example_name, neurasp_time, newrasp_time, slash_time=None):
    print(f"Old model time: {sum(elapsed_times[f'{example_name}_model'])}")
    if slash_time:
        print(f"SLASH model time: {sum(elapsed_times[f'slash_{example_name}_model'])}")
    print(f"New model time: {sum(elapsed_times[f'new_{example_name}_model'])}")
    print("==========")
    print(f"Old prob time: {sum(elapsed_times[f'{example_name}_prob'])}")
    if slash_time:
        print(f"SLASH prob time: {sum(elapsed_times[f'slash_{example_name}_prob'])}")
    print(f"New prob time: {sum(elapsed_times[f'new_{example_name}_prob'])}")
    print("==========")
    print(f"Old grad time: {sum(elapsed_times[f'{example_name}_grad'])}")
    if slash_time:
        print(f"SLASH grad time: {sum(elapsed_times[f'slash_{example_name}_grad'])}")
    print(f"New grad time: {sum(elapsed_times[f'new_{example_name}_grad'])}")
    print("==========")
    print(f"Total NeurASP time: {neurasp_time}")
    if slash_time:
        print(f"Total SLASH time: {slash_time}")
    print(f"Total new time: {newrasp_time}")


class TestSpeeds(unittest.TestCase):

    def test_speed_synthetic(self, num_models=1000, num_inputs=20, num_concepts=40):
        print(f"\nSynthetic speed test with {num_models} models, {num_inputs} inputs and {num_concepts} concepts.")
        models_new = torch.randint(0, num_concepts, (num_models, num_inputs))
        models = [[f"test(i{idx},{value})" for idx, value in enumerate(model)] for model in models_new]
        model_idx_list = [[(idx, value) for idx, value in enumerate(model)] for model in models_new]
        pc = [[f"test(i{image},{value})" for value in range(num_concepts)] for image in range(num_inputs)]
        parameters = np.random.random_sample((num_inputs, num_concepts)).tolist()
        selection_mask = torch.tensor([[True for _ in range(num_concepts)] for _ in range(num_inputs)])

        mock_return = (pc, parameters, False, "mock_asp", "mock_pi", "mock_remain_probs")
        mock_return_new = (pc, [torch.Tensor(parameter) for parameter in parameters], False, "mock_asp", "mock_pi",
                       "mock_remain_probs")

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

        with (mock.patch.object(MVPPNew, 'parse', return_value=mock_return_new),
              mock.patch.object(MVPPNew, 'normalize_probs')):
            mvpp_new = MVPPNew('')
            start_time = time.perf_counter()
            probs = mvpp_new.prob_of_interpretation(models_new)
            newrasp_prob_time = time.perf_counter() - start_time
            mvpp_new.mvppLearnRule(models_new, torch.Tensor(probs), 5)
            newrasp_grad_time = time.perf_counter() - newrasp_prob_time - start_time

        print(f"Old prob time: {neurasp_prob_time}")
        print(f"SLASH prob time: {slash_prob_time}")
        print(f"New prob time: {newrasp_prob_time}")
        print("==========")
        print(f"Old grad time: {neurasp_grad_time}")
        print(f"SLASH grad time: {slash_grad_time}")
        print(f"New grad time: {newrasp_grad_time}")

        assert (newrasp_prob_time + newrasp_grad_time < neurasp_prob_time + neurasp_grad_time)
        assert (newrasp_prob_time + newrasp_grad_time < slash_prob_time + slash_grad_time)

    def test_speeds_synthetic(self):
        """Test speeds of different implementations of probability and gradient calculations with synthetic data"""

        # 100 models with 10 inputs and 15 possible concepts
        self.test_speed_synthetic(100, 10, 15)

        # 1000 models with 18 inputs and 27 possible concepts
        self.test_speed_synthetic(1000, 18, 27)

        # 10000 models with 25 inputs and 40 possible concepts
        self.test_speed_synthetic(10000, 25, 40)

    def test_speeds_mnist_add(self):
        """Test speeds of different implementations for the MNIST Addition task"""
        os.chdir(os.path.abspath(ROOT_DIR + '/../examples/mnistAdd'))
        from examples.mnistAdd.dataGen import dataList, obsList
        from examples.mnistAdd.network import Net

        print("\nMNIST Add speed test")

        example_name = 'mnist_add'
        dprogram = ("img(i1). img(i2).\n"
                    "addition(A,B,N) :- digit(0,A,N1), digit(0,B,N2), N=N1+N2.\n"
                    "nn(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).")

        slash_program = ("img(i1). img(i2).\n"
                         "addition(A,B,N):- digit(0,+A,-N1), digit(0,+B,-N2), N=N1+N2, A!=B.\n"
                         "npp(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).")

        m = Net()
        nnMapping = {'digit': m}
        optimizers = {'digit': torch.optim.Adam(m.parameters())}

        # Sample 1000 examples
        dataList, obsList = sample_examples(dataList, obsList, 1000)

        # Original code
        neurasp_time = measure_neurasp_speed(dprogram, nnMapping, optimizers, dataList, obsList, example_name)

        # SLASH code
        dataList_slash = [{k: i.squeeze() for k, i in dataDict.items()} for dataDict in dataList]
        dataListLoader = torch.utils.data.DataLoader(list(zip(dataList_slash, obsList)))
        slash_time = measure_slash_speed(slash_program, nnMapping, optimizers, dataListLoader, example_name)

        # New code
        newrasp_time = measure_newrasp_speed(dprogram, nnMapping, optimizers, dataList, obsList, example_name)

        print_times(example_name, neurasp_time, newrasp_time, slash_time)

        # New code should be faster than existing code
        assert (newrasp_time < neurasp_time)
        assert (newrasp_time < slash_time)

    def test_speeds_top_k(self):
        """Test speeds of different implementations for the Top Knapsack task"""
        os.chdir(os.path.abspath(ROOT_DIR + '/../examples/top_k'))
        from examples.top_k.dataGen import dataList, obsList
        from examples.top_k.network import FC

        print("\nTop Knapsack speed test")

        example_name = 'top_k'
        dprogram = ("nn(in(10, k), [true, false]).\n"
                    "% define maxweight k\n"
                    "#const k = 7.\n"
                    ":- #sum{1, I : in(I,k,true)} > k.")

        m = FC(10, 50, 50, 50, 50, 50, 10)
        nnMapping = {'in': m}
        optimizers = {'in': torch.optim.Adam(m.parameters(), lr=0.001)}

        # Choose 1000 random examples
        dataList, obsList = sample_examples(dataList, obsList, 1000)

        # Original code
        neurasp_time = measure_neurasp_speed(dprogram, nnMapping, optimizers, dataList, obsList, example_name)

        # New code
        newrasp_time = measure_newrasp_speed(dprogram, nnMapping, optimizers, dataList, obsList, example_name)

        print_times(example_name, neurasp_time, newrasp_time)

        # New code should be faster than existing code
        assert (newrasp_time < neurasp_time)

    def test_speeds_add2x2(self):
        """Test speeds of different implementations for the Add 2x2 task"""
        os.chdir(os.path.abspath(ROOT_DIR + '/../examples/add2x2'))
        from examples.add2x2.dataGen import dataList, obsList
        from examples.add2x2.network import Net

        print("\nAdd 2x2 speed test")

        example_name = 'add2x2'
        dprogram = ("nn(digit(4,i), [0,1,2,3,4,5,6,7,8,9]).\n"
                    "add2x2(R1,R2,C1,C2) :- digit(0,i,N1), digit(1,i,N2), digit(2,i,N3), digit(3,i,N4), "
                    "R1=N1+N2, R2=N3+N4, C1=N1+N3, C2=N2+N4.")

        m = Net()
        nnMapping = {'digit': m}
        optimizers = {'digit': torch.optim.Adam(m.parameters())}

        # Choose 1000 random examples
        dataList, obsList = sample_examples(dataList, obsList, 1000)

        # Original code
        neurasp_time = measure_neurasp_speed(dprogram, nnMapping, optimizers, dataList, obsList, example_name)

        # New code
        newrasp_time = measure_newrasp_speed(dprogram, nnMapping, optimizers, dataList, obsList, example_name)

        print_times(example_name, neurasp_time, newrasp_time)

        # New code should be faster than existing code
        assert (newrasp_time < neurasp_time)

    def test_speeds_member(self):
        """Test speeds of different implementations for the Member task"""
        os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/../examples/member5')
        from examples.member5.dataGen import dataList, obsList
        from examples.member5.network import Net

        print("\nMember speed test")

        example_name = 'member'
        dprogram = ("nn(digit(5,i), [0,1,2,3,4,5,6,7,8,9]).\n"
                    "member(D,0) :- digit(0,i,N1), digit(1,i,N2), digit(2,i,N3), digit(3,i,N4), digit(4,i,N5),\n"
                    "check(D), D!=N1, D!=N2, D!=N3, D!=N4, D!=N5.\n"
                    "member(D,1) :- check(D), not member(D,0).")

        m = Net()
        nnMapping = {'digit': m}
        optimizers = {'digit': torch.optim.Adam(m.parameters())}

        # Choose 10 random examples
        dataList, obsList = sample_examples(dataList, obsList, 10)

        # Original code
        neurasp_time = measure_neurasp_speed(dprogram, nnMapping, optimizers, dataList, obsList, example_name)

        # New code
        newrasp_time = measure_newrasp_speed(dprogram, nnMapping, optimizers, dataList, obsList, example_name)

        print_times(example_name, neurasp_time, newrasp_time)

        # New code should be faster than existing code
        assert (newrasp_time < neurasp_time)

    def test_speeds_shortest_path(self):
        """Test speeds of different implementations for the Shortest Path task"""
        os.chdir(os.path.abspath(ROOT_DIR + '/../examples/shortest_path'))
        from examples.shortest_path.dataGen import dataList, obsList
        from examples.shortest_path.network import FC

        print("\nShortest path speed test")

        example_name = 'shortest_path'
        dprogram = ("nn(sp(24, g), [true, false]).\n"
                    "sp(X) :- sp(X,g,true).\n"
                    "sp(0,1) :- sp(0). sp(1,2) :- sp(1). sp(2,3) :- sp(2). sp(4,5) :- sp(3). sp(5,6) :- sp(4).\n"
                    "sp(6,7) :- sp(5). sp(8,9) :- sp(6). sp(9,10) :- sp(7). sp(10,11) :- sp(8). sp(12,13) :- sp(9).\n" 
                    "sp(13,14) :- sp(10). sp(14,15) :- sp(11). sp(0,4) :- sp(12). sp(4,8) :- sp(13).\n"
                    "sp(8,12) :- sp(14). sp(1,5) :- sp(15). sp(5,9) :- sp(16). sp(9,13) :- sp(17). sp(2,6) :- sp(18).\n"
                    "sp(6,10) :- sp(19). sp(10,14) :- sp(20). sp(3,7) :- sp(21). sp(7,11) :- sp(22).\n" 
                    "sp(11,15) :- sp(23). sp(X,Y) :- sp(Y,X).\n"
                    "mistake :- X=0..15, #count{Y: sp(X,Y)} = 1. mistake :- X=0..15, #count{Y: sp(X,Y)} >= 3.\n"
                    "reachable(X, Y) :- sp(X, Y). reachable(X, Y) :- reachable(X, Z), sp(Z, Y).\n"
                    "mistake :- sp(X, _), sp(Y, _), not reachable(X, Y).\n"
                    ":~ sp(X). [1, X]")

        m = FC(40, 50, 50, 50, 50, 50, 24)
        nnMapping = {'sp': m}
        optimizers = {'sp': torch.optim.Adam(m.parameters())}

        # Choose 500 random examples
        dataList, obsList = sample_examples(dataList, obsList, 500)

        # Original code
        neurasp_time = measure_neurasp_speed(dprogram, nnMapping, optimizers, dataList, obsList, example_name, opt=True)

        # New code
        newrasp_time = measure_newrasp_speed(dprogram, nnMapping, optimizers, dataList, obsList, example_name, opt=True)

        print_times(example_name, neurasp_time, newrasp_time)

        # New code should be faster than existing code
        assert (newrasp_time < neurasp_time)

    def test_speeds_follow_suit(self):
        """Test speeds of different implementations for Follow Suit task.
        WARNING: This test takes hours to complete!"""
        os.chdir(os.path.abspath(ROOT_DIR + '/../examples/follow_suit'))
        from examples.follow_suit.dataGen import dataList, obsList, facts, rules, dprogram
        from examples.follow_suit.network import Net

        print("\nFollow suit speed test")
        print("WARNING: This test typically takes hours to complete.")

        example_name = 'follow_suit'

        m = Net()
        nnMapping = {'card': m}
        optimizers = {'card': torch.optim.Adam(m.parameters())}

        # Choose 1 random example
        dataList, obsList = sample_examples(dataList, obsList, 1)

        # Original code
        neurasp_time = measure_neurasp_speed(dprogram, nnMapping, optimizers, dataList, obsList, example_name)

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
        slash_time = measure_slash_speed(slash_program, nnMapping, optimizers, dataListLoader, example_name)

        # New code
        newrasp_time = measure_newrasp_speed(dprogram, nnMapping, optimizers, dataList, obsList, example_name)

        print_times(example_name, neurasp_time, newrasp_time, slash_time)

        # New code should be faster than existing code
        assert (newrasp_time < neurasp_time)
        assert (newrasp_time < slash_time)

    def test_speeds_card_arithmetic(self):
        """Test speeds of different implementations for the card arithmetic task."""
        os.chdir(os.path.abspath(ROOT_DIR + '/../examples/card_arithmetic'))
        from examples.card_arithmetic.dataGen import get_dataset
        from examples.follow_suit.network import Net

        print("\nCard arithmetic speed test")
        example_name = 'card_arithmetic'

        m = Net()
        nnMapping = {'card': m}
        optimizers = {'card': torch.optim.Adam(m.parameters())}

        trainDataset, valDataset, dprogram = get_dataset('card_arithmetic_unique_3p', '../../data/playing_cards')

        dataList = []
        obsList = []

        # Choose 100 examples
        for idx, (data, obs) in enumerate(trainDataset):
            if idx >= 5:
                break
            dataList.append({'p': data['p'][0]})
            obsList.append(obs)

        # Original code
        neurasp_time = measure_neurasp_speed(dprogram, nnMapping, optimizers, dataList, obsList, example_name)

        # New code
        newrasp_time = measure_newrasp_speed(dprogram, nnMapping, optimizers, dataList, obsList, example_name)

        print_times(example_name, neurasp_time, newrasp_time)

        # New code should be faster than existing code
        assert (newrasp_time < neurasp_time)
