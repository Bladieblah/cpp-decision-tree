from typing import Any, List, Optional
import numpy as np
import numpy.typing as npt

class DecisionTreeCapsule: ...

def construct(
    criterion: Optional[int] = None,
    maxDepth: Optional[int] = None,
    minSamplesSplit: Optional[int] = None,
) -> DecisionTreeCapsule: ...

def fit(
    capsule: DecisionTreeCapsule,
    columnInds: npt.NDArray[np.uint32],
    rowPtr: npt.NDArray[np.uint32],
    data: npt.NDArray[np.double],
    numFeatures: int,
    yTrain: npt.NDArray[np.uint32],
    numClasses: int
) -> DecisionTreeCapsule: ...

def predict(
    capsule: DecisionTreeCapsule,
    columnInds: npt.NDArray[np.uint32],
    rowPtr: npt.NDArray[np.uint32],
    data: npt.NDArray[np.double],
    numFeatures: int,
) -> npt.NDArray[np.uint32]: ...

def predictNodes(
    capsule: DecisionTreeCapsule,
    column_inds: npt.NDArray[np.uint32],
    row_ptr: npt.NDArray[np.uint32],
    data: npt.NDArray[np.double],
    num_features: int,
) -> npt.NDArray[np.uint32]: ...

def predictProba(
    capsule: DecisionTreeCapsule,
    column_inds: npt.NDArray[np.uint32],
    row_ptr: npt.NDArray[np.uint32],
    data: npt.NDArray[np.double],
    num_features: int,
) -> npt.NDArray[np.uint32]: ...

def decisionPaths(
    capsule: DecisionTreeCapsule,
    column_inds: npt.NDArray[np.uint32],
    row_ptr: npt.NDArray[np.uint32],
    data: npt.NDArray[np.double],
    num_features: int,
) -> List[List[int]]: ...

def __getstate__(capsule: DecisionTreeCapsule) -> npt.NDArray[Any]: ...
def __setstate__(capsule: DecisionTreeCapsule, state: Any) -> DecisionTreeCapsule: ...
def deleteObject(capsule: DecisionTreeCapsule) -> None: ...
