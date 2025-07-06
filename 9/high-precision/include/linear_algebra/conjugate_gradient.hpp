#pragma once

#include "bailey/quad_double.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>
#include <chrono>

// ==============================================================================
//  共役勾配法（Conjugate Gradient Method）の実装
// ==============================================================================

namespace linear_algebra {

struct CGResult {
    int iter_final;
    bool is_converged;
    double time;
    std::vector<double> hist_relres_2;
    double true_relres_2;
    std::vector<double> hist_relerr_2;
    std::vector<double> hist_relerr_A;
};

inline CGResult conjugateGradient(
    const SpMat_QD& A, 
    const Vec_QD& b, 
    Vec_QD& x, 
    const Vec_QD& x_true,
    int max_iter = 1000, 
    double tolerance = 1e-15
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    Vec_QD r = b - A * x;
    Vec_QD p = r;
    QuadDouble rs_old = r.dot(r);
    
    QuadDouble b_norm = sqrt(b.dot(b));
    QuadDouble x_true_norm = sqrt(x_true.dot(x_true));
    
    CGResult result;
    result.hist_relres_2.reserve(max_iter + 1);
    result.hist_relerr_2.reserve(max_iter + 1);
    result.hist_relerr_A.reserve(max_iter + 1);
    
    // 初期値の履歴を記録
    QuadDouble init_residual_norm = sqrt(rs_old);
    result.hist_relres_2.push_back(to_double(init_residual_norm / b_norm));
    
    Vec_QD error = x - x_true;
    QuadDouble error_norm = sqrt(error.dot(error));
    result.hist_relerr_2.push_back(to_double(error_norm / x_true_norm));
    
    Vec_QD A_error = A * error;
    QuadDouble A_error_norm = sqrt(A_error.dot(A_error));
    result.hist_relerr_A.push_back(to_double(A_error_norm / x_true_norm));
    
    for (int k = 0; k < max_iter; ++k) {
        Vec_QD Ap = A * p;
        QuadDouble alpha = rs_old / p.dot(Ap);

        x = x + alpha * p;
        r = r - alpha * Ap;

        QuadDouble rs_new = r.dot(r);
        QuadDouble residual_norm = sqrt(rs_new);
        
        // 履歴記録
        result.hist_relres_2.push_back(to_double(residual_norm / b_norm));
        
        error = x - x_true;
        error_norm = sqrt(error.dot(error));
        result.hist_relerr_2.push_back(to_double(error_norm / x_true_norm));
        
        A_error = A * error;
        A_error_norm = sqrt(A_error.dot(A_error));
        result.hist_relerr_A.push_back(to_double(A_error_norm / x_true_norm));
        
        // 収束判定
        if (to_double(residual_norm) < tolerance) {
            result.iter_final = k + 1;
            result.is_converged = true;
            break;
        }

        p = r + (rs_new / rs_old) * p;
        rs_old = rs_new;
        
        if (k == max_iter - 1) {
            result.iter_final = max_iter;
            result.is_converged = false;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    result.time = duration.count() / 1000.0;
    
    // 真の相対残差を計算
    Vec_QD true_residual = A * x - b;
    QuadDouble true_residual_norm = sqrt(true_residual.dot(true_residual));
    result.true_relres_2 = to_double(true_residual_norm / b_norm);
    
    return result;
}

inline void print_num_results(const CGResult& results, const std::string& problem_name = "") {
    std::cout << "========================== " << std::endl;
    std::cout << "Numerical Results. " << std::endl;
    if (!problem_name.empty()) {
        std::cout << "Problem: " << problem_name << " " << std::endl;
    }
    std::cout << "========================== " << std::endl;

    if (results.is_converged) {
        std::cout << "Converged! (iter = " << results.iter_final << ")" << std::endl;
    } else {
        std::cout << "NOT converged. (max_iter = " << results.iter_final << ")" << std::endl;
    }

    std::cout << "# Iter.: " << results.iter_final << std::endl;
    std::cout << std::fixed << std::setprecision(3) << "Time[s]: " << results.time << std::endl;
    std::cout << std::scientific << std::setprecision(2);
    std::cout << "Relres_2norm = " << results.hist_relres_2[results.iter_final] << std::endl;
    std::cout << "True_Relres_2norm = " << results.true_relres_2 << std::endl;
    std::cout << "Relerr_2norm = " << results.hist_relerr_2[results.iter_final] << std::endl;
    std::cout << "Relerr_Anorm = " << results.hist_relerr_A[results.iter_final] << std::endl;
    std::cout << "========================== " << std::endl;
    std::cout << std::endl;
}

} // namespace linear_algebra