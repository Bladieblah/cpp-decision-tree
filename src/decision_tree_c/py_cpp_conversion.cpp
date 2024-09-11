#include <iostream>

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "py_cpp_conversion.hpp"
#include "decision_tree.hpp"

using namespace std;

#ifdef DEBUG
#define DebugLog(message) \
    do { \
        cerr << "DEBUG: " << message << endl; \
    } while (false)
#else
#define DebugLog(message) do {} while (false)
#endif

// ------------------- Helpers for converting between Python and C++ -------------------

PyArrayObject *checkNumpyArray(PyObject *obj) {
    // Exercise 2: Initial checks
    // Check for NULL
    if (obj == NULL) {
        PyErr_SetString(PyExc_ValueError, "Got NULL");
        return NULL;
    }

    // Check if the object is a numpy array
    if (!PyArray_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "Must be numpy ar");
        return NULL;
    }

    // Convert the object to a PyArrayObject * called `array`
    return (PyArrayObject *)obj;
}

template<typename T>
vector<T> numpyToVector(PyObject *obj) {
    vector<T> out;
    PyArrayObject *array = checkNumpyArray(obj);
    if (PyErr_Occurred() != NULL) return out;

    // Yay! You get this one for free, just uncomment the block
    PyArray_Descr* dtype = PyArray_DESCR(array);
    if (dtype->elsize != sizeof(T)) {
        PyErr_SetString(PyExc_ValueError, "Array dtype does not match the expected type.");
        return std::vector<T>();
    }

    // Exercise 3: Array reshaping
    // Flatten the array
    array = (PyArrayObject *)PyArray_Reshape(array, Py_BuildValue("I", PyArray_SIZE(array)));

    // Yay! You also get this one for free, just uncomment it
    size_t size = PyArray_DIM(array, 0);
    out.resize(size);

    for (size_t i = 0; i < size; i++) {
        out[i] = *reinterpret_cast<T*>(PyArray_GETPTR1(array, i));
    }

    return out;
}

template vector<double> numpyToVector<double>(PyObject *obj);
template vector<uint32_t> numpyToVector<uint32_t>(PyObject *obj);

vector<vector<uint32_t>> numpyToNestedVector(PyObject *obj) {
    vector<vector<uint32_t>> out;
    PyArrayObject *array = checkNumpyArray(obj);
    if (PyErr_Occurred() != NULL) return out;

    npy_intp ndim = PyArray_NDIM(array);
    if (ndim != 2) {
        PyErr_SetString(PyExc_ValueError, "Array must be 2D");
        return out;
    }

    // Determine array shape
    npy_intp size0 = PyArray_DIM(array, 0);
    npy_intp size1 = PyArray_DIM(array, 1);

    out.resize(size0);

    for (size_t i = 0; i < size0; i++) {
        vector<uint32_t> tmp(size1);
        for (size_t j = 0; j < size1; j++) {
            tmp[j] = (*reinterpret_cast<uint32_t*>(PyArray_GETPTR2(array, i, j)));
        }
        out[i] = tmp;
    }

    return out;
}

template<typename T>
T pythonToCpp(PyObject *obj);

template<>
uint32_t pythonToCpp<uint32_t>(PyObject *obj) {
    if (obj == NULL) {
        PyErr_SetString(PyExc_ValueError, "Unable to convert integer");
        return 0;
    }
    return PyLong_AsLong(obj);
}

template<>
double pythonToCpp<double>(PyObject *obj) {
    if (obj == NULL) {
        PyErr_SetString(PyExc_ValueError, "Unable to convert double");
        return 0;
    }
    return PyFloat_AsDouble(obj);
}

template<>
PyObject* pythonToCpp<PyObject*>(PyObject *obj) {
    if (obj == NULL) {
        PyErr_SetString(PyExc_ValueError, "Unable to convert object");
    }
    return obj;
}

template<typename T>
bool getDictItem(PyObject *dict, const char *key, T *value) {
    // Exercise 9: Dictionary item retrieval
    PyObject *dummy;
    // Use PyDict_GetItemWithError
    dummy = PyDict_GetItemWithError(dict, Py_BuildValue("s", key));

    // Add error checking logic
    if (dummy == NULL) {
        if (PyErr_Occurred() == NULL) {
            char msg[100];
            snprintf(msg, 99, "No key '%s' in dict", key);
            PyErr_SetString(PyExc_KeyError, msg);
        }
        return false;
    }

    *value = pythonToCpp<T>(dummy);
    if (PyErr_Occurred() != NULL) {
        return false;
    }

    return true;
}

template bool getDictItem(PyObject *dict, const char *key, uint32_t *value);
template bool getDictItem(PyObject *dict, const char *key, double *value);
template bool getDictItem(PyObject *dict, const char *key, PyObject **value);

vector<Node> arrayToNodes(PyArrayObject *array) {
    if (!PyArray_Check(array)) {
        PyErr_SetString(PyExc_TypeError, "Input must be a NumPy array.");
        return std::vector<Node>();
    }

    PyArray_Descr* dtype = PyArray_DESCR(array);
    if (dtype->elsize != sizeof(Node)) {
        PyErr_SetString(PyExc_ValueError, "Array dtype does not match the Node struct.");
        return std::vector<Node>();
    }

    PyArray_Descr *descr = PyArray_DTYPE(array);
    Node *nodes = static_cast<Node *>(PyArray_DATA(array));

    return vector<Node>(nodes, nodes + PyArray_SIZE(array));
}

PyObject *vectorToList(std::vector<uint32_t> data) {
    PyObject *list = PyList_New(data.size());

    for (size_t i = 0; i < data.size(); i++) {
        PyList_SET_ITEM(list, i, PyLong_FromLong(data[i]));
    }

    return list;
}

PyObject *nestedVectorToList(vector< vector<uint32_t> > data) {
    PyObject *list = PyList_New(data.size());

    for (size_t i = 0; i < data.size(); i++) {
        PyList_SET_ITEM(list, i, vectorToList(data[i]));
    }

return list;
}

PyObject *nestedVectorToArray(vector< vector<uint32_t> > data) {
    // Check for empty vector
    if (data.size() == 0) {
        npy_intp dims[] = {0, 0};
        return PyArray_SimpleNew(2, dims, NPY_UINT32);
    }

    // Exercise 7: 2D array creation
    // Determine the shape of the matrix
    npy_intp size0 = data.size();
    npy_intp size1 = data[0].size();
    
    // Create a C array of the correct size using `malloc`
    uint32_t *array = (uint32_t *)malloc(size0 * size1 * sizeof(uint32_t));
    
    // Fill the array using std::copy
    for (size_t i = 0; i < size0; i++) {
        copy(data[i].begin(), data[i].end(), array + size1 * i);
    }

    // Create a 2D numpy array using PyArray_SimpleNewFromData
    npy_intp shape[2] = {size0, size1};
    return PyArray_SimpleNewFromData(2, shape, NPY_UINT32, array);
}

PyObject *nodesToArray(vector<Node> nodeVector) {
    PyObject *op;
    PyArray_Descr *descr;

    // Exercise 8: Creating structured arrays
    // Step 1: Create a C array with the data in nodeVector
    Node *nodes = (Node *)malloc(nodeVector.size() * sizeof(Node));
    copy(nodeVector.begin(), nodeVector.end(), nodes);

    // Step 2: Create a list of tuples that represent the fields in the Node
    PyObject *dtype = Py_BuildValue("[(s, s), (s, s), (s, s), (s, s), (s, s), (s, s), (s, s)]",
        "left_child", "i4",
        "right_child", "i4",
        "feature", "i4",
        "threshold", "f8",
        "impurity", "f8",
        "n_samples", "i4",
        "label", "i4"
    );

    // I'll give you this one
    PyArray_DescrAlignConverter(op, &descr);
    // Uncomment this one after you've completed step 2
    Py_DECREF(dtype);

    // Create and return a structured numpy array using `PyArray_NewFromDescr`
    npy_intp shape[1] = {(npy_intp)nodeVector.size()};
    return PyArray_NewFromDescr(&PyArray_Type, descr, 1, shape, NULL, nodes, NULL, NULL);
}
