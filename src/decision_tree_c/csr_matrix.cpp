#include <vector>
#include <iostream>

#include "csr_matrix.hpp"

using namespace std;

CSRMatrix::CSRMatrix(
    vector<uint32_t> columnInds, 
    vector<uint32_t> rowPtr, 
    vector<double> data, 
    uint32_t numColumns
) {
    this->columnInds = columnInds;
    this->rowPtr = rowPtr;
    this->data = data;
    this->numColumns = numColumns;
}

size_t CSRMatrix::rowCount() {
    return this->rowPtr.size() - 1;
}

double CSRMatrix::get(size_t a, size_t b) {
    size_t c = rowPtr[a];
    size_t d = rowPtr[a + 1];

    for (size_t i = c; i < d; i++) {
        if (columnInds[i] == b) {
            return data[i];
        }
    }

    return 0;
}
