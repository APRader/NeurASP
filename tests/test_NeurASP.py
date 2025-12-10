import unittest
import unittest.mock as mock
import numpy as np
import torch

from mvpp import MVPP
from mvpp_new import MVPP as MVPPNew
from mvpp_slash import MVPP as MVPPSlash

from neurasp import NeurASP
from newrasp import NeurASP as NewrASP

class TestNeurASP(unittest.TestCase):

    def test_prob_of_interpretation(self):
        """Test that probabilities are calculated correctly"""

        # 6 models, 9 images
        Is = [
            ['test(i1,3)', 'test(i2,0)', 'test(i3,3)', 'test(i4,1)', 'test(i5,3)', 'test(i6,3)', 'test(i7,3)',
             'test(i8,0)', 'test(i9,3)'],
            ['test(i1,0)', 'test(i2,3)', 'test(i3,1)', 'test(i4,1)', 'test(i5,2)', 'test(i6,2)', 'test(i7,0)',
             'test(i8,1)', 'test(i9,3)'],
            ['test(i1,0)', 'test(i2,2)', 'test(i3,1)', 'test(i4,1)', 'test(i5,2)', 'test(i6,1)', 'test(i7,2)',
             'test(i8,0)', 'test(i9,3)'],
            ['test(i1,0)', 'test(i2,3)', 'test(i3,1)', 'test(i4,0)', 'test(i5,2)', 'test(i6,1)', 'test(i7,1)',
             'test(i8,0)', 'test(i9,1)'],
            ['test(i1,0)', 'test(i2,3)', 'test(i3,3)', 'test(i4,0)', 'test(i5,0)', 'test(i6,2)', 'test(i7,1)',
             'test(i8,1)', 'test(i9,2)'],
            ['test(i1,1)', 'test(i2,3)', 'test(i3,3)', 'test(i4,0)', 'test(i5,0)', 'test(i6,2)', 'test(i7,2)',
             'test(i8,1)', 'test(i9,0)']
        ]
        Is_new = [[3, 0, 3, 1, 3, 3, 3, 0, 3], [0, 3, 1, 1, 2, 2, 0, 1, 3], [0, 2, 1, 1, 2, 1, 2, 0, 3],
                  [0, 3, 1, 0, 2, 1, 1, 0, 1], [0, 3, 3, 0, 0, 2, 1, 1, 2], [1, 3, 3, 0, 0, 2, 2, 1, 0]]
        model_idx_list = [[(0, 3), (1, 0), (2, 3), (3, 1), (4, 3), (5, 3), (6, 3), (7, 0), (8, 3)],
                          [(0, 0), (1, 3), (2, 1), (3, 1), (4, 2), (5, 2), (6, 0), (7, 1), (8, 3)],
                          [(0, 0), (1, 2), (2, 1), (3, 1), (4, 2), (5, 1), (6, 2), (7, 0), (8, 3)],
                          [(0, 0), (1, 3), (2, 1), (3, 0), (4, 2), (5, 1), (6, 1), (7, 0), (8, 1)],
                          [(0, 0), (1, 3), (2, 3), (3, 0), (4, 0), (5, 2), (6, 1), (7, 1), (8, 2)],
                          [(0, 1), (1, 3), (2, 3), (3, 0), (4, 0), (5, 2), (6, 2), (7, 1), (8, 0)]]

        # 4 outputs per image
        pc = [
            ['test(i1,0)', 'test(i1,1)', 'test(i1,2)', 'test(i1,3)'],
            ['test(i2,0)', 'test(i2,1)', 'test(i2,2)', 'test(i2,3)'],
            ['test(i3,0)', 'test(i3,1)', 'test(i3,2)', 'test(i3,3)'],
            ['test(i4,0)', 'test(i4,1)', 'test(i4,2)', 'test(i4,3)'],
            ['test(i5,0)', 'test(i5,1)', 'test(i5,2)', 'test(i5,3)'],
            ['test(i6,0)', 'test(i6,1)', 'test(i6,2)', 'test(i6,3)'],
            ['test(i7,0)', 'test(i7,1)', 'test(i7,2)', 'test(i7,3)'],
            ['test(i8,0)', 'test(i8,1)', 'test(i8,2)', 'test(i8,3)'],
            ['test(i9,0)', 'test(i9,1)', 'test(i9,2)', 'test(i9,3)']
        ]

        parameters = [[0.2, 0.1, 0.8, 0.7], [0.9, 0.9, 0.4, 0.3], [0.6, 0.3, 0.1, 0.1], [0.7, 0.1, 0.2, 0.4],
                     [0.1,  0.7, 0.5, 0.3], [0, 0.5, 0.9, 0.1], [0.5, 0.7, 0, 0.7], [0.6, 0.3, 0.1, 0.7],
                     [0.9, 0, 0.2, 0.5]]

        mock_return = (pc, parameters, False, "mock_asp", "mock_pi", "mock_remain_probs")
        mock_return_new = (pc, [torch.Tensor(parameter) for parameter in parameters], False, "mock_asp", "mock_pi",
                           "mock_remain_probs")

        probs = [0.00003969, 0.00006075, 0, 0, 0.000015876, 0]
        neurasp_probs = []
        slash_probs = []

        with (mock.patch.object(MVPP, 'parse', return_value = mock_return),
              mock.patch.object(MVPP, 'normalize_probs'),
              mock.patch.object(MVPPNew, 'parse', return_value=mock_return_new),
              mock.patch.object(MVPPNew, 'normalize_probs'),
              mock.patch.object(MVPPSlash, 'parse', return_value=mock_return + ([],))):
            mvpp = MVPP('')
            mvpp_slash = MVPPSlash('')
            mvpp_slash.M = parameters
            mvpp_new = MVPPNew('')
            for i in range(len(Is)):
                neurasp_probs.append(mvpp.prob_of_interpretation(Is[i]))
                slash_probs.append((mvpp_slash.prob_of_interpretation(Is[i], model_idx_list[i])))
            new_probs = mvpp_new.prob_of_interpretation(Is_new)

        np.testing.assert_almost_equal(neurasp_probs, probs)
        np.testing.assert_almost_equal(slash_probs, probs)
        np.testing.assert_almost_equal(new_probs, probs)

    def test_mvppLearnRule(self):
        """Test that gradients are calculated correctly"""

        # 8 models, 9 images
        models = [
            ['test(i1,1)', 'test(i2,0)', 'test(i3,0)', 'test(i4,5)', 'test(i5,7)', 'test(i6,7)', 'test(i7,5)',
             'test(i8,3)', 'test(i9,2)'],
            ['test(i1,6)', 'test(i2,2)', 'test(i3,0)', 'test(i4,1)', 'test(i5,0)', 'test(i6,6)', 'test(i7,4)',
             'test(i8,5)', 'test(i9,8)'],
            ['test(i1,5)', 'test(i2,1)', 'test(i3,8)', 'test(i4,6)', 'test(i5,1)', 'test(i6,2)', 'test(i7,4)',
             'test(i8,6)', 'test(i9,4)'],
            ['test(i1,8)', 'test(i2,8)', 'test(i3,3)', 'test(i4,0)', 'test(i5,7)', 'test(i6,0)', 'test(i7,3)',
             'test(i8,1)', 'test(i9,3)'],
            ['test(i1,2)', 'test(i2,4)', 'test(i3,7)', 'test(i4,1)', 'test(i5,3)', 'test(i6,3)', 'test(i7,5)',
             'test(i8,1)', 'test(i9,2)'],
            ['test(i1,3)', 'test(i2,3)', 'test(i3,5)', 'test(i4,3)', 'test(i5,8)', 'test(i6,8)', 'test(i7,3)',
             'test(i8,0)', 'test(i9,2)'],
            ['test(i1,3)', 'test(i2,4)', 'test(i3,8)', 'test(i4,5)', 'test(i5,8)', 'test(i6,5)', 'test(i7,2)',
             'test(i8,0)', 'test(i9,4)'],
            ['test(i1,7)', 'test(i2,4)', 'test(i3,3)', 'test(i4,1)', 'test(i5,0)', 'test(i6,2)', 'test(i7,5)',
             'test(i8,7)', 'test(i9,6)']
        ]
        models_new = torch.IntTensor([[1, 0, 0, 5, 7, 7, 5, 3, 2], [6, 2, 0, 1, 0, 6, 4, 5, 8], [5, 1, 8, 6, 1, 2, 4, 6, 4],
                      [8, 8, 3, 0, 7, 0, 3, 1, 3], [2, 4, 7, 1, 3, 3, 5, 1, 2], [3, 3, 5, 3, 8, 8, 3, 0, 2],
                      [3, 4, 8, 5, 8, 5, 2, 0, 4], [7, 4, 3, 1, 0, 2, 5, 7, 6]])
        model_idx_list = [[(0, 1), (1, 0), (2, 0), (3, 5), (4, 7), (5, 7), (6, 5), (7, 3), (8, 2)],
                          [(0, 6), (1, 2), (2, 0), (3, 1), (4, 0), (5, 6), (6, 4), (7, 5), (8, 8)],
                          [(0, 5), (1, 1), (2, 8), (3, 6), (4, 1), (5, 2), (6, 4), (7, 6), (8, 4)],
                          [(0, 8), (1, 8), (2, 3), (3, 0), (4, 7), (5, 0), (6, 3), (7, 1), (8, 3)],
                          [(0, 2), (1, 4), (2, 7), (3, 1), (4, 3), (5, 3), (6, 5), (7, 1), (8, 2)],
                          [(0, 3), (1, 3), (2, 5), (3, 3), (4, 8), (5, 8), (6, 3), (7, 0), (8, 2)],
                          [(0, 3), (1, 4), (2, 8), (3, 5), (4, 8), (5, 5), (6, 2), (7, 0), (8, 4)],
                          [(0, 7), (1, 4), (2, 3), (3, 1), (4, 0), (5, 2), (6, 5), (7, 7), (8, 6)]]

        # 9 outputs per image
        pc = [
            ['test(i1,0)', 'test(i1,1)', 'test(i1,2)', 'test(i1,3)', 'test(i1,4)', 'test(i1,5)', 'test(i1,6)',
             'test(i1,7)', 'test(i1,8)'],
            ['test(i2,0)', 'test(i2,1)', 'test(i2,2)', 'test(i2,3)', 'test(i2,4)', 'test(i2,5)', 'test(i2,6)',
             'test(i2,7)', 'test(i2,8)'],
            ['test(i3,0)', 'test(i3,1)', 'test(i3,2)', 'test(i3,3)', 'test(i3,4)', 'test(i3,5)', 'test(i3,6)',
             'test(i3,7)', 'test(i3,8)'],
            ['test(i4,0)', 'test(i4,1)', 'test(i4,2)', 'test(i4,3)', 'test(i4,4)', 'test(i4,5)', 'test(i4,6)',
             'test(i4,7)', 'test(i4,8)'],
            ['test(i5,0)', 'test(i5,1)', 'test(i5,2)', 'test(i5,3)', 'test(i5,4)', 'test(i5,5)', 'test(i5,6)',
             'test(i5,7)', 'test(i5,8)'],
            ['test(i6,0)', 'test(i6,1)', 'test(i6,2)', 'test(i6,3)', 'test(i6,4)', 'test(i6,5)', 'test(i6,6)',
             'test(i6,7)', 'test(i6,8)'],
            ['test(i7,0)', 'test(i7,1)', 'test(i7,2)', 'test(i7,3)', 'test(i7,4)', 'test(i7,5)', 'test(i7,6)',
             'test(i7,7)', 'test(i7,8)'],
            ['test(i8,0)', 'test(i8,1)', 'test(i8,2)', 'test(i8,3)', 'test(i8,4)', 'test(i8,5)', 'test(i8,6)',
             'test(i8,7)', 'test(i8,8)'],
            ['test(i9,0)', 'test(i9,1)', 'test(i9,2)', 'test(i9,3)', 'test(i9,4)', 'test(i9,5)', 'test(i9,6)',
             'test(i9,7)', 'test(i9,8)']
        ]

        probs = np.array([0, 0.1, 0, 1, 0.1, 0.6, 0, 0.2])

        parameters = [[0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1], [0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1],
                    [0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1], [0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.3],
                    [0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1],
                    [0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0, 0.1], [0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3],
                    [0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2]]

        selection_mask = torch.tensor([[True for _ in range(9)] for _ in range(9)])

        mock_return = (pc, parameters, False, "mock_asp", "mock_pi", "mock_remain_probs")
        mock_return_new = (pc, [torch.Tensor(parameter) for parameter in parameters], False, "mock_asp", "mock_pi",
                           "mock_remain_probs")


        grads = [[-9, -9, -8.5, -3, -9, -9, -8.5, -8, 1], [-10, -10, -9, -4, -7, -10, -10, -10, 0],
                 [-7.25, -8.25, -8.25, 3.75, -8.25, -5.25, -8.25, -7.75, -8.25], [1, -7, -9, -3, -9, -9, -9, -9, -9],
                 [-7.75, -9.25, -9.25, -8.25, -9.25, -9.25, -9.25, 0.75, -3.25],
                 [0.5, -9.5, -7.5, -9, -9.5, -9.5, -9, -9.5, -3.5],
                 [-9, -9, -9, 7, -8.5, -7.5, -9, -9, -9], [-3.5, 1.5, -9.5, -9.5, -9.5, -8.5, -9.5, -8.5, -9.5],
                 [-9.75, -9.75, -2.75, 0.25, -9.75, -9.75, -7.75, -9.75, -9.25]]
        neurasp_grads = []

        with (mock.patch.object(MVPP, 'parse', return_value=mock_return),
              mock.patch.object(MVPP, 'normalize_probs'),
              mock.patch.object(MVPPNew, 'parse', return_value=mock_return_new),
              mock.patch.object(MVPPNew, 'normalize_probs'),
              mock.patch.object(MVPPSlash, 'parse', return_value=mock_return + ([],))):
            mvpp = MVPP('')
            mvpp_slash = MVPPSlash('')
            mvpp_slash.max_n = 9
            mvpp_slash.M = torch.tensor(parameters)
            mvpp_slash.binary_rule_belongings = {}
            mvpp_slash.selection_mask = selection_mask
            mvpp_new = MVPPNew('')
            for ruleIdx in range(9):
                neurasp_grads.append(mvpp.mvppLearnRule(ruleIdx, models, probs))
            slash_grads = mvpp_slash.mvppLearnRule(models, model_idx_list, 'cpu', torch.tensor(probs))
            new_grads = mvpp_new.mvppLearnRule(models_new, torch.Tensor(probs), 9)

        np.testing.assert_almost_equal(neurasp_grads, grads)
        np.testing.assert_almost_equal(slash_grads, grads)
        np.testing.assert_almost_equal(new_grads, grads)

    def test_find_k_SM_under_obs(self):
        """Test that models are found correctly"""

        # 5 concepts, 5 choices
        pi_prime = ("1{test(1,i1,1); test(1,i1,2); test(1,i1,3); test(1,i1,4); test(1,i1,5)}1.\n"
                    "1{test(1,i2,1); test(1,i2,2); test(1,i2,3); test(1,i2,4); test(1,i2,5)}1.\n"
                    "1{test(1,i3,1); test(1,i3,2); test(1,i3,3); test(1,i3,4); test(1,i3,5)}1.\n"
                    "1{test(1,i4,1); test(1,i4,2); test(1,i4,3); test(1,i4,4); test(1,i4,5)}1.\n"
                    "1{test(1,i5,1); test(1,i5,2); test(1,i5,3); test(1,i5,4); test(1,i5,5)}1.\n"
                    "result(N) :- test(1,i1,N1), test(1,i2,N2), test(1,i3,N3), test(1,i4,N4), test(1,i5,N5), "
                    "N=(N1+N3)*10+N2+N4-N5.")
        obs = ":- not result(109)."
        pc = {'test/3:i1': ['1','2','3','4','5'], 'test/3:i2': ['1','2','3','4','5'],
              'test/3:i3': ['1','2','3','4','5'], 'test/3:i4': ['1','2','3','4','5'],
              'test/3:i5': ['1','2','3','4','5']}

        # There is one stable model that satisfies the observation
        models = ['result(109)', 'test(1,i1,5)', 'test(1,i2,5)', 'test(1,i3,5)', 'test(1,i4,5)', 'test(1,i5,1)']
        models_new = np.array([[4,4,4,4,0]])

        mock_return = (pc, [], False, "mock_asp", pi_prime, "mock_remain_probs")

        # Test NeurASP
        with (mock.patch.object(MVPP, 'parse', return_value=mock_return),
              mock.patch.object(MVPP, 'normalize_probs')):
            mvpp = MVPP('')
            neurasp_models = mvpp.find_k_SM_under_obs(obs)
            assert len(neurasp_models) == 1
            assert sorted(neurasp_models[0]) == models

        # Test SLASH
        with mock.patch.object(MVPPSlash, 'parse', return_value=mock_return + ([],)):
            mvpp_slash = MVPPSlash('')
            slash_models = mvpp_slash.find_k_SM_under_query(obs)
            assert len(slash_models) == 1
            assert sorted(slash_models[0]) == models

        # Test new implementation
        with (mock.patch.object(MVPPNew, 'parse', return_value=mock_return),
              mock.patch.object(MVPPNew, 'normalize_probs')):
            mvpp_new = MVPPNew('')
            new_models = mvpp_new.find_k_SM_under_obs(obs)
            assert (new_models == models_new).all()

    def test_find_all_opt_SM_under_obs_WC(self):
        """Test that optimal models are found correctly"""

        # 2 concepts, 2 choices
        pi_prime = ("1{in(1,i1,true); in(1,i1,false)}1. 1{in(1,i2,true); in(1,i2,false)}1."
                    ":- #sum{1, I : in(1,I,true)} > 1.")
        obs = ":~ in(1,i1,true). [-3,r1] :~ in(1,i2,true). [-6,r2]"
        pc = {'in/3:i1': ['true', 'false'], 'in/3:i2': ['true', 'false']}

        # There is one optimal stable model that minimises the weak constraints
        models = ['in(1,i1,false)', 'in(1,i2,true)']
        models_new = np.array([[1, 0]])

        mock_return = (pc, [], False, "mock_asp", pi_prime, "mock_remain_probs")

        # Test NeurASP
        with (mock.patch.object(MVPP, 'parse', return_value=mock_return),
              mock.patch.object(MVPP, 'normalize_probs')):
            mvpp = MVPP('')
            neurasp_models = mvpp.find_all_opt_SM_under_obs_WC(obs)
            assert len(neurasp_models) == 1
            assert sorted(neurasp_models[0]) == models

        # Test SLASH
        with mock.patch.object(MVPPSlash, 'parse', return_value=mock_return + ([],)):
            mvpp_slash = MVPPSlash('')
            slash_models = mvpp_slash.find_all_opt_SM_under_query_WC(obs)
            assert len(slash_models) == 1
            assert sorted(slash_models[0]) == models

        # Test new implementation
        with (mock.patch.object(MVPPNew, 'parse', return_value=mock_return),
              mock.patch.object(MVPPNew, 'normalize_probs')):
            mvpp_new = MVPPNew('')
            new_models = mvpp_new.find_k_SM_under_obs(obs, opt=True)
            assert (new_models == models_new).all()

    def test_stable_model_caching(self):
        """Test that stable models are stored correctly"""

        # Create a simple neural network
        m = torch.nn.Linear(9,8)
        nnMapping = {'test': m}
        optimizer = {'test': torch.optim.Adam(m.parameters())}

        # 9 data points, each with 8 choices
        dataList = [{'i': torch.rand(9)} for i in range(9)]
        obsList = [':- not obs(8).',':- not obs(4).', ':-not obs(8).', ':- not obs(1).', ':- not obs(3)',
                   ':- not obs(2)',':- nots obs(2)', ':- nots obs(3).', ':- not obs(6).']
        dataset = list(zip(dataList, obsList))
        mock_return = ('program', 'program_pr', 'program_asp')
        stable_models = ['8', '4', '8', '1', '3', '2', '2', '3', '6']

        # Test NeurASP
        with (mock.patch.object(NeurASP, 'parse', return_value=mock_return),
              mock.patch('neurasp.MVPP') as mock_mvpp):
            # Return the number n in :- not obs(n) when calling the stable model function
            mock_mvpp.return_value.find_k_SM_under_obs = lambda obs, k: obs.split('(')[1].split(')')[0]
            NeurASPobj = NeurASP('dprogram', nnMapping, optimizer)
            NeurASPobj.learn(dataList, obsList, 2, storeSM=True)
            assert NeurASPobj.stableModels == stable_models

        # Test new implementation
        with (mock.patch.object(NewrASP, 'parse', return_value=mock_return),
              mock.patch('newrasp.MVPP') as mock_mvpp):
            mock_mvpp.return_value.find_k_SM_under_obs = lambda obs, k, opt: obs.split('(')[1].split(')')[0]
            NewrASPobj = NewrASP('dprogram', nnMapping, optimizer)
            NewrASPobj.learn(dataset, 2, storeSM=True)
            assert NewrASPobj.stableModels == {obs: sm for obs, sm in zip(obsList, stable_models)}

    def test_downstream_accuracy(self):
        """Test that downstream accuracy is calculated correctly"""
        # Set seed so that nn output is the same every time
        # Predictions will be [1,1,1,1,2,1,1,1] and [1,1,1,2,1,0,1,1]
        torch.manual_seed(8)

        # 2 data points, each with 8 concepts and 3 choices
        m = torch.nn.Linear(3, 3)
        dataset = [({'i': torch.rand(8, 3)}, ':- not obs(1).'), ({'i': torch.rand(8, 3)}, ':- not obs(3).')]

        nnMapping = {'test': m}
        optimizer = {'test': torch.optim.Adam(m.parameters())}
        mock_return = ('program', 'program_pr', 'program_asp')
        nn_prob = [[('test', 0, 'i', 0)], [('test', 1, 'i', 0)], [('test', 2, 'i', 0)], [('test', 3, 'i', 0)],
                   [('test', 4, 'i', 0)], [('test', 5, 'i', 0)], [('test', 6, 'i', 0)], [('test', 7, 'i', 0)]]

        # Only obs(1) has cached stable models
        # No entry matches the nn prediction
        stable_models = {':- not obs(1).': torch.IntTensor([[2,2,2,2,2,2,0,2], [2,0,2,1,1,1,1,1],
                                                            [1,1,1,1,0,0,2,1], [2,0,1,2,1,1,0,0]])}

        # The stable model for obs(3) has to be fetched from the dmvpp
        # The sixth entry matches the nn prediction
        found_stable_models = torch.IntTensor([[0,1,1,2,2,2,0,0],[2,2,2,0,1,0,1,2],[0,1,2,0,2,2,2,2],[2,0,0,0,0,2,1,1],
                               [2,2,0,0,0,0,2,2],[1,1,1,2,1,0,1,1],[0,0,2,0,0,1,2,0],[0,1,1,1,1,2,0,2],
                               [1,2,1,0,2,0,1,0]])
        def mvpp_side_effect(obs,k,opt):
            if obs == ':- not obs(3).':
                return found_stable_models
            return None

        with (mock.patch.object(NewrASP, 'parse', return_value=mock_return),
              mock.patch('newrasp.MVPP') as mock_mvpp):
            mock_mvpp.parameters = [0,0,0,0,0,0,0,0]
            mock_mvpp.find_k_SM_under_obs.side_effect = mvpp_side_effect
            NewrASPobj = NewrASP('dprogram', nnMapping, optimizer)
            NewrASPobj.nnOutputs = {'test': ['i']}
            NewrASPobj.stableModels = stable_models
            NewrASPobj.mvpp['nnPrRuleNum'] = 8
            NewrASPobj.mvpp['nnProb'] = nn_prob
            accuracy = NewrASPobj.downstream_accuracy(dataset, mock_mvpp, True, False)

            # Only the second out of the two inputs yields the correct prediction
            assert accuracy == 0.5
