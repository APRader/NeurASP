import unittest
import unittest.mock as mock
import numpy as np

from mvpp import MVPP

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

        probs = [0.00003969, 0.00006075, 0, 0, 0.000015876, 0]
        old_probs = []

        with (mock.patch.object(MVPP, 'parse', return_value = mock_return),
              mock.patch.object(MVPP, 'normalize_probs')):
            mvpp = MVPP('')
            for I in Is:
                old_probs.append(mvpp.prob_of_interpretation(I))
            new_probs = mvpp.prob_of_interpretation_new(Is_new)

        np.testing.assert_almost_equal(old_probs, probs)
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
        models_new = [[1, 0, 0, 5, 7, 7, 5, 3, 2], [6, 2, 0, 1, 0, 6, 4, 5, 8], [5, 1, 8, 6, 1, 2, 4, 6, 4],
                      [8, 8, 3, 0, 7, 0, 3, 1, 3], [2, 4, 7, 1, 3, 3, 5, 1, 2], [3, 3, 5, 3, 8, 8, 3, 0, 2],
                      [3, 4, 8, 5, 8, 5, 2, 0, 4], [7, 4, 3, 1, 0, 2, 5, 7, 6]]

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

        mock_return = (pc, parameters, False, "mock_asp", "mock_pi", "mock_remain_probs")

        grads = [[-9, -9, -8.5, -3, -9, -9, -8.5, -8, 1], [-10, -10, -9, -4, -7, -10, -10, -10, 0],
                 [-7.25, -8.25, -8.25, 3.75, -8.25, -5.25, -8.25, -7.75, -8.25], [1, -7, -9, -3, -9, -9, -9, -9, -9],
                 [-7.75, -9.25, -9.25, -8.25, -9.25, -9.25, -9.25, 0.75, -3.25],
                 [0.5, -9.5, -7.5, -9, -9.5, -9.5, -9, -9.5, -3.5],
                 [-9, -9, -9, 7, -8.5, -7.5, -9, -9, -9], [-3.5, 1.5, -9.5, -9.5, -9.5, -8.5, -9.5, -8.5, -9.5],
                 [-9.75, -9.75, -2.75, 0.25, -9.75, -9.75, -7.75, -9.75, -9.25]]
        old_grads = []

        with (mock.patch.object(MVPP, 'parse', return_value=mock_return),
              mock.patch.object(MVPP, 'normalize_probs')):
            mvpp = MVPP('')
            for ruleIdx in range(9):
                old_grads.append(mvpp.mvppLearnRule(ruleIdx, models, probs))
            new_grads = mvpp.mvppLearnRuleNew(models_new, np.array(probs), 9)

        np.testing.assert_almost_equal(old_grads, grads)
        np.testing.assert_almost_equal(new_grads, grads)