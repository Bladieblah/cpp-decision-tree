from enum import Enum
from pprint import pprint
from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix

import DecisionTreeC


class Criterion(Enum):
    Gini = 0
    Entropy = 1


class DecisionTree:
    tree_ = None
    classes: npt.NDArray[Any]
    n_classes: int
    is_fitted: bool = False

    def __init__(
        self,
        criterion: Optional[Criterion|int] = None,
        max_depth: Optional[int] = None,
        min_samples_split: Optional[int] = None,
    ):
        if isinstance(criterion, Criterion):
            criterion = criterion.value

        kwargs: Dict[str, Any] = {
            "criterion": criterion,
            "maxDepth": max_depth,
            "minSamplesSplit": min_samples_split,
        }

        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.tree_ = DecisionTreeC.construct(**kwargs)
    
    def fit(
        self,
        x_train: csr_matrix,
        y_train: npt.NDArray[Any]
    ):
        if self.tree_ is None:
            raise Exception("No capsule found")
        
        self.classes, y_encoded = np.unique(y_train, return_inverse=True)
        self.n_classes = self.classes.shape[0]
        
        self.tree_ = DecisionTreeC.fit(
            self.tree_,
            x_train.indices.astype(np.uint32),
            x_train.indptr.astype(np.uint32),
            x_train.data.astype(np.double),
            x_train.shape[1],
            y_encoded.astype(np.uint32),
            self.n_classes
        )

        self.is_fitted = True

        return self
    
    def predict(self, x_predict: csr_matrix):
        if self.tree_ is None:
            raise Exception("No capsule found")
        
        if not self.is_fitted:
            raise Exception("Classifier is not fitted")

        class_indices = DecisionTreeC.predict(
            self.tree_,
            x_predict.indices.astype(np.uint32),
            x_predict.indptr.astype(np.uint32),
            x_predict.data.astype(np.double),
            x_predict.shape[1]
        )

        return self.classes.take(class_indices)
    
    def predict_nodes(self, x_predict: csr_matrix):
        if self.tree_ is None:
            raise Exception("No capsule found")
        
        if not self.is_fitted:
            raise Exception("Classifier is not fitted")

        node_indices = DecisionTreeC.predictNodes(
            self.tree_,
            x_predict.indices.astype(np.uint32),
            x_predict.indptr.astype(np.uint32),
            x_predict.data.astype(np.double),
            x_predict.shape[1]
        )

        return node_indices
    
    def predict_proba(self, x_predict: csr_matrix):
        if self.tree_ is None:
            raise Exception("No capsule found")
        
        if not self.is_fitted:
            raise Exception("Classifier is not fitted")

        probabilities = DecisionTreeC.predictProba(
            self.tree_,
            x_predict.indices.astype(np.uint32),
            x_predict.indptr.astype(np.uint32),
            x_predict.data.astype(np.double),
            x_predict.shape[1]
        )

        return probabilities / np.sum(probabilities, axis=1)[:,None]
    
    def decision_paths(self, x_predict: csr_matrix):
        if self.tree_ is None:
            raise Exception("No capsule found")
        
        if not self.is_fitted:
            raise Exception("Classifier is not fitted")

        paths = DecisionTreeC.decisionPaths(
            self.tree_,
            x_predict.indices.astype(np.uint32),
            x_predict.indptr.astype(np.uint32),
            x_predict.data.astype(np.double),
            x_predict.shape[1]
        )

        return paths
    
    def __getstate__(self):
        if self.tree_ is None:
            raise Exception("No capsule found")

        return DecisionTreeC.__getstate__(self.tree_)

    def __setstate__(self, state: Any):
        if self.tree_ is None:
            raise Exception("No capsule found")

        self.tree_ = DecisionTreeC.__setstate__(self.tree_, state)

    def __del__(self):
        if self.tree_:
            DecisionTreeC.deleteObject(self.tree_)
            del self.tree_
