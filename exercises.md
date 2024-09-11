# Useful links
- Format strings for [PyArg_Parsetuple](https://docs.python.org/3/c-api/arg.html)
- Storing objects in [capsules](https://docs.python.org/3/c-api/capsule.html)
- Conversion functions
    - [Integers](https://docs.python.org/3/c-api/long.html)
    - [Floats](https://docs.python.org/3/c-api/float.html)
    - [Strings](https://docs.python.org/3/c-api/unicode.html)
    - [Lists](https://docs.python.org/3/c-api/list.html)
    - [Dictionaries](https://docs.python.org/3/c-api/dict.html)
- [Exception handling](https://docs.python.org/3/c-api/exceptions.html)
- [Numpy array API](https://numpy.org/doc/stable/reference/c-api/array.html)
- How to use [malloc](https://en.cppreference.com/w/c/memory/malloc)


# Exercises

## Exercise 1
Finish the `construct` function in [decision_tree_wrapper.cpp](/src/decision_tree_c/decision_tree_wrapper.cpp):
- Instantiate a `DecisionTree` object
    - Use the `new` keyword to create a pointer to a DecisionTree.
- Create a capsule using `PyCapsule_New`
    - Make sure to give it the name `treePtr` or other parts of the code will break
    - You can give `NULL` for the 3rd argument
- Return the capsule using `Py_BuildValue`

## Exercise 2
Add 2 checks to the `checkNumpyArray` function in [py_cpp_conversion.cpp](/src/decision_tree_c/py_cpp_conversion.cpp)
- First check if `obj` is a nullptr. In this case raise a `ValueError` using PyErr_SetString and return the `out` vector
- Add a second check if `obj` is a numpy array, using `PyArray_Check`. In this case raise a `TypeError` and return the `out` vector
- Convert `obj` to a `PyArrayObject` and return it

## Exercise 3
Finish the `numpyToVector` function:
- Flatten the array to 1D using [PyArray_Reshape](https://numpy.org/doc/stable/reference/c-api/array.html#shape-manipulation) (hint: construct the shape using `Py_BuildValue`)
- Initialize `out` to the correct size using its resize method, get the correct size using [PyArray_SIZE](https://numpy.org/doc/stable/reference/c-api/array.html#array-structure-and-data-access)

## Exercise 4
You're now ready to finish the fitting function! 
- Use `PyArg_ParseTuple` to parse the function arguments. Refer to [DecisionTreeC.pyi](/src/decision_tree/DecisionTreeC.pyi) for the function signature.
- Fill the remaining 4 vectors using your shiny new `numpyToVector` conversion function. Make sure you use the correct type when you call it.
    - columnInds
    - rowPtr
    - data
    - yTrain
- Fetch the `DecisionTree` pointer using [PyCapsule_GetPointer](https://docs.python.org/3/c-api/capsule.html)
- Fit the tree by calling the `fit` method

## Exercise 5
This one's a doozy, you're going to finish the `callPredictionFunction` function! Fortunately, you can copy-paste a bunch of code from the `fit` function you just finished.
- Declare the required variables
- Parse the arguments
- Convert numpy arrays to c++ vectors
- Instantiate tree and CSRMatrix
All this simply requires copying part of the `fit` function and deleting unnecessary parts.

**If you managed this, you just got all 4 prediction methods to work!**

In the remaining exercises, we will implement `__getstate__` and `__setstate__` functionalities for persisting the decision tree.

## Exercise 6
We'll start simple. First, in the `getstate` function in [decision_tree_wrapper.cpp](/src/decision_tree_c/decision_tree_wrapper.cpp), store the 5 integer attributes of the decision tree in a dictionary. It should be of the form
```python
{
    "criterion": <tree->getCriterion()>,
    "max_depth": <tree->getMaxDepth()>,
    "min_samples_split": <tree->getMinSamplesSplit()>,
    "num_classes": <tree->getNumClasses()>,
    "num_features": <tree->getNumFeatures()>,
}
```
Look up the proper conversion functions yourself in the official python documentation. Don't forget to replace the return statement and return the dictionary you created!

## Exercise 7
The values array `tree->getValues()` is a rectangular matrix. Convert it to a 2D numpy array by implementing the `nestedVectorToArray` function in [decision_tree_wrapper.cpp](/src/decision_tree_c/decision_tree_wrapper.cpp)

1. Determine the dimensions of the matrix
2. Create a 1 dimensional C array of type `uint32_t` using `malloc`
3. Copy all the data from the `values` vector to the C array using `std::copy`
4. Create a new 2D numpy array using `PyArray_SimpleNewFromData`
6. Return the array

Don't forget to add the array to the dict in `getstate`, with key `values`.

## Exercise 8
Now for a tough one, you will implement `nodesToArray`. This function converts the vector of `Node` structs (see [](/include/decision_tree.hpp)) to a numpy [structured array](https://numpy.org/doc/stable/user/basics.rec.html) with the same fields.

1. Convert the `nodeVector` to a C array, similar to steps 1-3 in exercise 7
2. Create the dtype of the structured array using `Py_BuildValue`
    - Each tuple is a pair of `("<fieldname>", "<type>")`
    - Each field in `Node` should have a corresponding field in the dtype
    - You can use the `s` format specifier to insert strings in the tuple
    - Make sure the widths of the fields match (hint: a `double` is 8 bits so it will be "f4", a `uint32_t` is 4 bits).
3. Use [PyArray_NewFromDescr](https://numpy.org/doc/stable/reference/c-api/array.html#creating-arrays) to create the structured array
    - For the first argument give it the type `&PyArray_Type`
    - The `strides`, `flags` and `obj` arguments can be null

Don't forget to add the array to the dict in `getstate`, with key `nodes`.

## Exercise 9
Implement the `getDictItem` functionality in [decision_tree_wrapper.cpp](/src/decision_tree_c/decision_tree_wrapper.cpp). It should return `true` if successful, otherwise `false`
- Use `PyDict_GetItemWithError` to retrieve the item
    - You need to convert the `const char *key` to a python string
- In case the item did not exist in the dict, raise an error and return `false`
    - This error should be equivalent to `KeyError("No key '%s' in dict" % key)` in python
    - Hint: Format a string using [snprintf](https://cplusplus.com/reference/cstdio/snprintf/)
- If a different kind of exception occurred, an exception is already set so you can simply return `false`
