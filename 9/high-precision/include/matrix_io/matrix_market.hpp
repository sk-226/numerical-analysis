#pragma once

#include "bailey/quad_double.hpp"
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <string>
#include <vector>
#include <fstream>

// ==============================================================================
//  Matrix Market形式ファイルの読み込み機能
// ==============================================================================

namespace matrix_io {

inline SpMat_QD loadMatrixMarket(const std::string& filename) {
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        fast_matrix_market::matrix_market_header header;
        std::vector<int> rows, cols;
        std::vector<double> values;
        
        // ストリームから読み込み
        fast_matrix_market::read_matrix_market_triplet(
            file, header, rows, cols, values
        );

        SpMat_QD mat(header.nrows, header.ncols);
        std::vector<Eigen::Triplet<QuadDouble>> triplets;
        triplets.reserve(values.size());


        for (size_t i = 0; i < values.size(); ++i) {
            // fast_matrix_marketは0-indexedで返す
            int row = rows[i];
            int col = cols[i];
            
            // 範囲チェック
            if (row < 0 || row >= static_cast<int>(header.nrows) || 
                col < 0 || col >= static_cast<int>(header.ncols)) {
                throw std::runtime_error("Index out of bounds: row=" + std::to_string(row) + 
                                       ", col=" + std::to_string(col) + 
                                       " for matrix " + std::to_string(header.nrows) + 
                                       "x" + std::to_string(header.ncols));
            }
            
            triplets.emplace_back(row, col, QuadDouble(values[i]));
        }

        mat.setFromTriplets(triplets.begin(), triplets.end());
        return mat;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Matrix Market read error: " + std::string(e.what()) + " for file: " + filename);
    }
}

} // namespace matrix_io