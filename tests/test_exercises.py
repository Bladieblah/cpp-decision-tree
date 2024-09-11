import unittest

import numpy as np
import sklearn.metrics as skm

from scipy.sparse import csr_matrix
from sklearn.datasets import load_iris

import DecisionTreeC
from decision_tree.decision_tree import Criterion, DecisionTree


class CheckExercises(unittest.TestCase):
    def _get_data(self):
        X, y = load_iris(return_X_y = True)
        return csr_matrix(X), y
    
    def test_exercise_1(self):
        tree = DecisionTreeC.construct(Criterion.Entropy.value, 4, 5)
        self.assertEqual(str(tree.__class__), "<class 'PyCapsule'>", "Function did not return a PyCapsule")

        criterion, maxdepth, mss, _ , _ = DecisionTreeC.checkTree(tree)

        self.assertEqual(criterion, Criterion.Entropy.value, "You did not properly set the `criterion` property in the DecisionTree constructor")
        self.assertEqual(maxdepth, 4, "You did not properly set the `maxDepth` property in the DecisionTree constructor")
        self.assertEqual(mss, 5, "You did not properly set the `minSamplesSplit` property in the DecisionTree constructor")

    def test_exercise_2(self):
        with self.assertRaises(ValueError):
            DecisionTreeC.checkNumpyToVector1()

        with self.assertRaises(TypeError):
            DecisionTreeC.checkNumpyToVector2()
    
    def test_exercise_3(self):
        ar = np.array([1], dtype=np.uint16);
        with self.assertRaises(ValueError):
            DecisionTreeC.checkNumpyToVector3(ar)
        
        ar = np.array([[1, 2], [3, 4]], dtype=np.double)
        self.assertEqual(DecisionTreeC.checkNumpyToVector3(ar), 4)
    
    def test_exercise_4(self):
        tree = DecisionTreeC.construct()

        ar = np.array([1], dtype=np.uint32)
        result = DecisionTreeC.fit(tree, ar, ar, ar.astype(np.double), 1, ar, 1)
        self.assertEqual(str(result.__class__), "<class 'PyCapsule'>", "Function did not return a PyCapsule")

        _, _, _, numNodes, numValues = DecisionTreeC.checkTree(tree)
        self.assertEqual(numNodes, 1, "The tree does not appear properly fitted, incorrect number of nodes")
        self.assertEqual(numValues, 1, "The tree does not appear properly fitted, incorrect number of values")

        X, y = self._get_data()
        dt = DecisionTree()
        dt.fit(X, y)
        _, _, _, numNodes, numValues = DecisionTreeC.checkTree(dt.tree_)
        self.assertGreater(numNodes, 200, "The tree does not appear properly fitted, incorrect number of nodes")
        self.assertGreater(numValues, 200, "The tree does not appear properly fitted, incorrect number of values")

    def test_exercise_5(self):
        X, y = self._get_data()
        dt = DecisionTree()
        dt.fit(X, y)

        predictions = dt.predict(X)
        self.assertGreater(skm.accuracy_score(y, predictions), 0.95, "Predictions do not match expected values")
        
        nodes = dt.predict_nodes(X)
        self.assertEqual(nodes.shape[0], predictions.shape[0])
        
        proba = dt.predict_proba(X)
        self.assertEqual(proba.shape[0], predictions.shape[0])

        paths = dt.decision_paths(X)
        self.assertEqual(len(paths), predictions.shape[0])
    
    def test_exercise_6(self):
        X, y = self._get_data()
        dt = DecisionTree(0, 1000, 2)
        dt.fit(X, y)

        state = dt.__getstate__()
        self.assertIsInstance(state, dict, "State is not a dict")

        self.assertTrue("criterion" in state, "`criterion` not in state")
        self.assertEqual(state["criterion"], 0, "`criterion` has incorrect value")

        self.assertTrue("max_depth" in state, "`max_depth` not in state")
        self.assertEqual(state["max_depth"], 1000, "`max_depth` has incorrect value")

        self.assertTrue("min_samples_split" in state, "`min_samples_split` not in state")
        self.assertEqual(state["min_samples_split"], 2, "`min_samples_split` has incorrect value")

        self.assertTrue("num_classes" in state, "`num_classes` not in state")
        self.assertEqual(state["num_classes"], 3, "`num_classes` has incorrect value")

        self.assertTrue("num_features" in state, "`num_features` not in state")
        self.assertEqual(state["num_features"], 4, "`num_features` has incorrect value")
    
    def test_exercise_7(self):
        X, y = self._get_data()
        dt = DecisionTree(0, 1000, 2)
        dt.fit(X, y)

        state = dt.__getstate__()
        self.assertIsInstance(state, dict, "State is not a dict")
        self.assertTrue("values" in state, "`values` not in state")

        values = state["values"]
        self.assertIsInstance(values, np.ndarray, "Values is not a numpy array")
        self.assertEqual(len(values.shape), 2, "Values has incorrect shape")
        self.assertEqual(values.shape[1], 3, "Values has incorrect shape")
        self.assertEqual(values.dtype, np.uint32, "Values ahas incorrect dtype")
    
    def test_exercise_8(self):
        X, y = self._get_data()
        dt = DecisionTree(0, 1000, 2)
        dt.fit(X, y)

        state = dt.__getstate__()
        self.assertIsInstance(state, dict, "State is not a dict")
        self.assertTrue("nodes" in state, "`nodes` not in state")
        
        nodes = state["nodes"]
        self.assertEqual(len(nodes.shape), 1, "`nodes` has incorrect shape")
        self.assertEqual(len(nodes.dtype.names), 7, "`nodes` dtype has incorrect number of fields")
        
        descr = [x for x in nodes.dtype.descr if x[0] != ""]
        self.assertEqual(descr[0], ('left_child', '<i4'))
        self.assertEqual(descr[1], ('right_child', '<i4'))
        self.assertEqual(descr[2], ('feature', '<i4'))
        self.assertEqual(descr[3], ('threshold', '<f8'))
        self.assertEqual(descr[4], ('impurity', '<f8'))
        self.assertEqual(descr[5], ('n_samples', '<i4'))
        self.assertEqual(descr[6], ('label', '<i4'))

    def test_exercise_9(self):
        X, y = self._get_data()
        dt = DecisionTree(0, 1000, 2)
        dt.fit(X, y)
        state = dt.__getstate__()

        dt2 = DecisionTree()
        dt2.__setstate__(state)

        state2 = dt2.__getstate__()

        for key in ["criterion", "max_depth", "min_samples_split", "num_classes", "num_features"]:
            self.assertEqual(state[key], state2[key], f"state['{key}'] mismatch")
        
        self.assertTrue(np.all(state["values"] == state2["values"]), "state['values'] mismatch")
        self.assertTrue(np.all(state["nodes"] == state2["nodes"]), "state['nodes'] mismatch")


def suite():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(CheckExercises)
    return suite
