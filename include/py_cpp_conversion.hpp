#ifndef PY_CPP_CONVERSION_H
#define PY_CPP_CONVERSION_H

#include <string>
#include <vector>

#include "Python.h"
#include "decision_tree.hpp"

// ------------------- Helpers for converting between Python and C++ -------------------

// ------------------- Python -> C++ -------------------

template<typename T>
std::vector<T> numpyToVector(PyObject *array);

template<typename T>
bool getDictItem(PyObject *dict, const char *key, T *value);

std::vector<Node> arrayToNodes(PyObject *array);

// ------------------- C++ -> Python -------------------

PyObject *vectorToList(std::vector<uint32_t> data);

PyObject *nestedVectorToList(std::vector< std::vector<uint32_t> > data);

PyObject *nodesToArray(std::vector<Node> nodeVector);

PyObject *nestedVectorToArray(std::vector< std::vector<uint32_t> > data);

#endif
