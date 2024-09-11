#include <stdio.h>
#include <iostream>
#include <functional>

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "csr_matrix.hpp"
#include "decision_tree.hpp"

using namespace std;

#ifdef DEBUG
#define DebugLog(message) \
    do { \
        cerr << "DEBUG: " << message << endl; \
    } while (false)
#else
#define DebugLog(message) 
#endif

// Include the cpp file directly, otherwise it segfaults
#include "py_cpp_conversion.cpp"


#ifdef DEBUG
#define DebugLog(message) \
    do { \
        cerr << "DEBUG: " << message << endl; \
    } while (false)
#else
#define DebugLog(message) 
#endif

template<typename T>
T callPredictionFunction(PyObject *args, function<T(DecisionTree*, CSRMatrix)> callMethod) {
    DecisionTree *tree;
    CSRMatrix xPredict;

    // Copypaste the required parts of `fit` below here:
    vector<uint32_t> columnInds;
    vector<uint32_t> rowPtr;
    vector<double> data;
    uint32_t numFeatures;

    PyObject *columnArray, *rowArray, *dataArray;
    PyObject *treeCapsule;

    // Exercise 4: Finishing the fitting method.
    // Use PyArg_ParseTuple to parse the arguments
    PyArg_ParseTuple(args, "OOOOI", 
        &treeCapsule,
        &columnArray,
        &rowArray,
        &dataArray,
        &numFeatures
    );

    columnInds = numpyToVector<uint32_t>(columnArray);
    // Fill the remainging 3 vectors using numpyToVector
    rowPtr = numpyToVector<uint32_t>(rowArray);
    data = numpyToVector<double>(dataArray);

    // Fetch the DecisionTree pointer
    tree = (DecisionTree *)PyCapsule_GetPointer(treeCapsule, "treePtr");
    xPredict = CSRMatrix(columnInds, rowPtr, data, numFeatures);


    if (xPredict.numColumns != tree->getNumFeatures()) {
        PyErr_SetString(PyExc_Exception, "Number of features in matrix does not match training data");
        return T();
    }

    T predictions = callMethod(tree, xPredict);

    return predictions;
}

template vector<uint32_t> callPredictionFunction<vector<uint32_t>>(PyObject *args, function<vector<uint32_t>(DecisionTree*, CSRMatrix)> callMethod);
template Matrix callPredictionFunction<Matrix>(PyObject *args, function<Matrix(DecisionTree*, CSRMatrix)> callMethod);
template vector<vector<uint32_t>> callPredictionFunction<vector<vector<uint32_t>>>(PyObject *args, function<vector<vector<uint32_t>>(DecisionTree*, CSRMatrix)> callMethod);

// ------------------- Class wrappers -------------------

PyObject *construct(PyObject *self, PyObject *args, PyObject *keywds) {
    DebugLog("Constructing tree");
    uint32_t criterion = ImpurityType::Gini, maxDepth = UINT32_MAX, minSamplesSplit = 2;

    static char *kwlist[] = {"criterion", "maxDepth", "minSamplesSplit", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|III", kwlist, &criterion, &maxDepth, &minSamplesSplit)) {
        return NULL;
    }

    // Exercise 1: constructing the tree
    // Instantiate a DecisionTree
    DecisionTree *tree = new DecisionTree(criterion, maxDepth, minSamplesSplit);
    
    // Create a pycapsule with a reference to the tree using PyCapsule_New
    PyObject *capsule = PyCapsule_New((void *)tree, "treePtr", NULL);

    // Return the capsule
    return capsule;
    Py_RETURN_NONE; // Replace this return statement!
}

PyObject *fit(PyObject *self, PyObject *args) {
    DebugLog("Fitting tree");
    vector<uint32_t> columnInds;
    vector<uint32_t> rowPtr;
    vector<double> data;
    uint32_t numClasses, numFeatures;
    vector<uint32_t> yTrain;

    PyObject *columnArray, *rowArray, *dataArray, *yArray;
    PyObject *treeCapsule;

    // Exercise 4: Finishing the fitting method.
    // Use PyArg_ParseTuple to parse the arguments
    PyArg_ParseTuple(args, "OOOOIOI", 
        &treeCapsule,
        &columnArray,
        &rowArray,
        &dataArray,
        &numFeatures,
        &yArray,
        &numClasses
    );

    columnInds = numpyToVector<uint32_t>(columnArray);
    // Fill the remainging 3 vectors using numpyToVector
    rowPtr = numpyToVector<uint32_t>(rowArray);
    data = numpyToVector<double>(dataArray);
    yTrain = numpyToVector<uint32_t>(yArray);

    if (PyErr_Occurred() != NULL) return NULL;

    // Fetch the DecisionTree pointer
    DecisionTree *tree = (DecisionTree *)PyCapsule_GetPointer(treeCapsule, "treePtr");
    CSRMatrix xTrain(columnInds, rowPtr, data, numFeatures);

    // Fit the tree
    tree->fit(xTrain, yTrain, numClasses);
    
    return Py_BuildValue("O", treeCapsule);
    // Py_RETURN_NONE;
}

auto callPredict = [](DecisionTree *tree, CSRMatrix xPredict){return tree->predict(xPredict);};
auto callPredictNodes = [](DecisionTree *tree, CSRMatrix xPredict){return tree->predictNodes(xPredict);};
auto callPredictProba = [](DecisionTree *tree, CSRMatrix xPredict){return tree->predictProba(xPredict);};
auto callDecisionPaths = [](DecisionTree *tree, CSRMatrix xPredict){return tree->decisionPaths(xPredict);};

PyObject *predict(PyObject *self, PyObject *args) {
    DebugLog("Predicting classes");
    vector<uint32_t> predictions = callPredictionFunction<vector<uint32_t>>(args, callPredict);

    uint32_t *array = (uint32_t *)malloc(predictions.size() * sizeof(uint32_t));
    copy(predictions.begin(), predictions.end(), array);

    npy_intp dims[1] = {static_cast<npy_intp>(predictions.size())};
    return PyArray_SimpleNewFromData(1, dims, NPY_UINT32, array);
}

PyObject *predictNodes(PyObject *self, PyObject *args) {
    DebugLog("Predicting nodes");
    vector<uint32_t> predictions = callPredictionFunction<vector<uint32_t>>(args, callPredictNodes);

    uint32_t *array = (uint32_t *)malloc(predictions.size() * sizeof(uint32_t));
    copy(predictions.begin(), predictions.end(), array);

    npy_intp dims[1] = {static_cast<npy_intp>(predictions.size())};
    return PyArray_SimpleNewFromData(1, dims, NPY_UINT32, array);
}

PyObject *predictProba(PyObject *self, PyObject *args) {
    DebugLog("Predicting probabilities");
    Matrix probabilities = callPredictionFunction<Matrix>(args, callPredictProba);
    uint32_t *array = (uint32_t *)malloc(probabilities.data.size() * sizeof(uint32_t));
    copy(probabilities.data.begin(), probabilities.data.end(), array);

    npy_intp dims[1] = {static_cast<npy_intp>(probabilities.data.size())};
    PyObject *npArray = PyArray_SimpleNewFromData(1, dims, NPY_UINT32, array);
    PyObject *shape = Py_BuildValue("ii", probabilities.h, probabilities.w);
    npArray = PyArray_Reshape((PyArrayObject *)npArray, shape);

    return npArray;
}

PyObject *decisionPaths(PyObject *self, PyObject *args) {
    DebugLog("Predicting decision paths");
    vector< vector<uint32_t> > paths = callPredictionFunction<vector<vector<uint32_t>>>(args, callDecisionPaths);

    return nestedVectorToList(paths);
}

PyObject *getstate(PyObject *self, PyObject *args) {
    DebugLog("__getstate__");
    PyObject *treeCapsule;
    PyArg_ParseTuple(args, "O", &treeCapsule);
    DecisionTree *tree = (DecisionTree *)PyCapsule_GetPointer(treeCapsule, "treePtr");

    // Exercise 6: __getstate__ part 1 (integer attributes)
    // Create a dict matching the one in exercises.md
    PyObject *dict = PyDict_New();

    PyDict_SetItemString(dict, "criterion", PyLong_FromLong(tree->getCriterion()));
    PyDict_SetItemString(dict, "max_depth", PyLong_FromLong(tree->getMaxDepth()));
    PyDict_SetItemString(dict, "min_samples_split", PyLong_FromLong(tree->getMinSamplesSplit()));
    PyDict_SetItemString(dict, "num_classes", PyLong_FromLong(tree->getNumClasses()));
    PyDict_SetItemString(dict, "num_features", PyLong_FromLong(tree->getNumFeatures()));
    
    // Exercise 7: __getstate__ part 2 (values)
    PyDict_SetItemString(dict, "values", nestedVectorToArray(tree->getValues()));
    
    // Exercise 8: __getstate__ part 3 (nodes)
    PyDict_SetItemString(dict, "nodes", nodesToArray(tree->getNodes()));

    return dict;
}

PyObject *setstate(PyObject *self, PyObject *args) {
    DebugLog("__setstate__");
    PyObject *treeCapsule, *state;
    PyArg_ParseTuple(args, "OO", &treeCapsule, &state);
    DecisionTree *tree = (DecisionTree *)PyCapsule_GetPointer(treeCapsule, "treePtr");

    uint32_t value;
    if (!getDictItem(state, "criterion", &value)) return NULL; else tree->setCriterion(value);
    if (!getDictItem(state, "max_depth", &value)) return NULL; else tree->setMaxDepth(value);
    if (!getDictItem(state, "min_samples_split", &value)) return NULL; else tree->setMinSamplesSplit(value);
    if (!getDictItem(state, "num_classes", &value)) return NULL; else tree->setNumClasses(value);
    if (!getDictItem(state, "num_features", &value)) return NULL; else tree->setNumFeatures(value);

    PyObject *obj;

    if (!getDictItem(state, "nodes", &obj)) return NULL;
    vector<Node> nodes = arrayToNodes((PyArrayObject *)obj);
    if (PyErr_Occurred() != NULL) return NULL; else tree->setNodes(nodes);
    
    if (!getDictItem(state, "values", &obj)) return NULL;
    vector<vector<uint32_t>> values = numpyToNestedVector(obj);
    if (PyErr_Occurred() != NULL) return NULL; else tree->setValues(values);

    return Py_BuildValue("O", treeCapsule);
}

PyObject *deleteObject(PyObject *self, PyObject *args) {
    DebugLog("__del__");
    PyObject *treeCapsule;
    PyArg_ParseTuple(args, "O", &treeCapsule);
    
    DecisionTree *tree = (DecisionTree *)PyCapsule_GetPointer(treeCapsule, "treePtr");
    delete tree;

    Py_RETURN_NONE;
}

PyObject *checkTree(PyObject *self, PyObject *args) {
    DebugLog("checkTree");
    PyObject *treeCapsule;
    PyArg_ParseTuple(args, "O", &treeCapsule);
    
    DecisionTree *tree = (DecisionTree *)PyCapsule_GetPointer(treeCapsule, "treePtr");

    return Py_BuildValue("IIIII", tree->getCriterion(), tree->getMaxDepth(), tree->getMinSamplesSplit(), tree->getNodes().size(), tree->getValues().size());
}

PyObject *checkNumpyToVector1(PyObject *self, PyObject *args) {
    numpyToVector<double>(NULL);
    return NULL;
}

PyObject *checkNumpyToVector2(PyObject *self, PyObject *args) {
    numpyToVector<double>(Py_None);
    return NULL;
}

PyObject *checkNumpyToVector3(PyObject *self, PyObject *args) {
    PyObject *obj;
    PyArg_ParseTuple(args, "O", &obj);

    vector<double> result = numpyToVector<double>(obj);
    if (PyErr_Occurred() != NULL) return NULL;
    
    return Py_BuildValue("I", (result.size()));
}

PyMethodDef DecisionTreeCFunctions[] = {
    {"construct",
    (PyCFunction)(void(*)(void))construct, METH_VARARGS | METH_KEYWORDS,
    "Create `DecisionTree` object"},
    
    {"fit",
    fit, METH_VARARGS,
    "Fit the classifier"},
    
    {"predict",
    predict, METH_VARARGS,
    "Predict data"},
    
    {"predictNodes",
    predictNodes, METH_VARARGS,
    "Predict get node ids of predictions"},
    
    {"predictProba",
    predictProba, METH_VARARGS,
    "Predict data with probabilities"},
    
    {"decisionPaths",
    decisionPaths, METH_VARARGS,
    "Get paths of predictions"},
    
    {"__getstate__",
    getstate, METH_VARARGS,
    "Get current state"},
    
    {"__setstate__",
    setstate, METH_VARARGS,
    "Set current state"},
    
    {"deleteObject",
    deleteObject, METH_VARARGS,
    "Delete `DecisionTree` object"},

    {"checkTree",
    checkTree, METH_VARARGS,
    "Just a check nothing to see here"},

    {"checkNumpyToVector1",
    checkNumpyToVector1, METH_VARARGS,
    "Just a check nothing to see here"},

    {"checkNumpyToVector2",
    checkNumpyToVector2, METH_VARARGS,
    "Just a check nothing to see here"},

    {"checkNumpyToVector3",
    checkNumpyToVector3, METH_VARARGS,
    "Just a check nothing to see here"},

    {NULL, NULL, 0, NULL}      // Last function description must be empty.
                               // Otherwise, it will create seg fault while
                               // importing the module.
};


struct PyModuleDef DecisionTreeCModule = {
/*
 *  Structure which defines the module.
 *
 *  For more info look at: https://docs.python.org/3/c-api/module.html
 *
 */
    PyModuleDef_HEAD_INIT,
    "DecisionTreeC",
    // Docstring for the module.
    "Decision Tree class in C++", 
    -1,// Used by sub-interpreters, if you do not know what it is then you do not need it, keep -1 .
    DecisionTreeCFunctions
};


PyMODINIT_FUNC PyInit_DecisionTreeC(void) {
    import_array();
    return PyModule_Create(&DecisionTreeCModule);
}
