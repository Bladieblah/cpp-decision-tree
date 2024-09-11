#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H

#include <vector>

class CSRMatrix {
public:
    CSRMatrix() {};
    CSRMatrix(
        std::vector<uint32_t> columnInds,
        std::vector<uint32_t> rowPtr,
        std::vector<double> data,
        uint32_t numColumns
    );

    double get(size_t a, size_t b);
    size_t rowCount();

    std::vector<uint32_t> columnInds; // Contains nonzero column indices for each row
    std::vector<uint32_t> rowPtr; // Indicates location of each row in columnInds
    std::vector<double> data;

    size_t numColumns; // Number of columns in matrix
};

#endif