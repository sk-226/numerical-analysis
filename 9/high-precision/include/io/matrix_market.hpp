#pragma once

#include "bailey/precision_traits.hpp"
#include <fstream>
#include <stdexcept>
#include <vector>
#include <sstream>
#include <Eigen/Sparse>

namespace io {

template<typename T>
typename bailey::PrecisionTraits<T>::matrix_type
loadMatrixMarket(const std::string& filename) {
    using Traits = bailey::PrecisionTraits<T>;
    using MatrixType = typename Traits::matrix_type;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Matrix Market file not found: " + filename);
    }
    
    std::string line;
    // Read header line
    if (!std::getline(file, line) || line.find("%%MatrixMarket") != 0) {
        throw std::runtime_error("Invalid Matrix Market file format");
    }
    
    // Check if matrix is symmetric
    bool is_symmetric = line.find("symmetric") != std::string::npos;
    
    // Skip comment lines
    while (std::getline(file, line) && line[0] == '%') {}
    
    // Read dimensions and nnz
    std::istringstream iss(line);
    int nrows, ncols, nnz;
    if (!(iss >> nrows >> ncols >> nnz)) {
        throw std::runtime_error("Failed to read matrix dimensions");
    }
    
    std::vector<Eigen::Triplet<T>> triplets;
    // Reserve more space for symmetric matrices (need to store both triangles)
    triplets.reserve(is_symmetric ? 2 * nnz : nnz);
    
    // Read matrix entries
    for (int i = 0; i < nnz; ++i) {
        if (!std::getline(file, line)) {
            throw std::runtime_error("Unexpected end of file");
        }
        
        std::istringstream entry_iss(line);
        int row, col;
        double value;
        if (!(entry_iss >> row >> col >> value)) {
            throw std::runtime_error("Failed to parse matrix entry");
        }
        
        // Convert to 0-indexed
        row -= 1;
        col -= 1;
        
        triplets.emplace_back(row, col, T(value));
        
        // For symmetric matrices, add the symmetric entry if not on diagonal
        if (is_symmetric && row != col) {
            triplets.emplace_back(col, row, T(value));
        }
    }
    
    MatrixType matrix(nrows, ncols);
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    
    return matrix;
}

// Template specialization helper for file path construction
inline std::string constructMatrixPath(const std::string& matrix_name, 
                                     const std::string& base_dir = "/work/inputs") {
    return base_dir + "/" + matrix_name + ".mtx";
}

} // namespace io