#pragma once

#include "bailey/precision_traits.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <variant>

namespace algorithms {

/// Results structure for Conjugate Gradient solver
/// Contains convergence history and performance metrics
template<typename T>
struct CGResult {
    using Traits = bailey::PrecisionTraits<T>;
    
    // Convergence information
    int iterations_performed;           ///< Number of iterations performed
    bool converged;                     ///< Whether convergence was achieved
    double computation_time;            ///< Wall-clock time in seconds
    
    // Convergence history
    std::vector<double> hist_relres_2;  ///< Relative residual 2-norm history
    std::vector<double> hist_relerr_2;  ///< Relative error 2-norm history  
    std::vector<double> hist_relerr_A;  ///< Relative error A-norm history
    
    // Final metrics
    double true_relres_2;               ///< True relative residual (b-Ax verification)
    double final_residual_norm;         ///< Final relative residual norm
    double initial_residual_norm;       ///< Initial residual norm
    std::string precision_name{Traits::name()};  ///< Precision level name
};

/// Conjugate Gradient solver with comprehensive convergence tracking
/// 
/// Solves the linear system Ax = b using the Conjugate Gradient method.
/// Supports multiple precision levels through template specialization.
/// 
/// @param A Symmetric positive definite matrix
/// @param b Right-hand side vector  
/// @param x Initial guess (modified in-place)
/// @param x_true True solution for error analysis
/// @param max_iter Maximum number of iterations
/// @param tolerance Convergence tolerance for relative residual
/// @return CGResult containing convergence history and statistics
template<typename T>
CGResult<T> conjugateGradient(
    const typename bailey::PrecisionTraits<T>::matrix_type& A, 
    const typename bailey::PrecisionTraits<T>::vector_type& b, 
    typename bailey::PrecisionTraits<T>::vector_type& x, 
    const typename bailey::PrecisionTraits<T>::vector_type& x_true,
    int max_iter, 
    double tolerance
) {
    using Traits = bailey::PrecisionTraits<T>;
    using VectorType = typename Traits::vector_type;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Precompute norms for relative error calculations
    T norm2_b = sqrt(b.dot(b));
    T norm2_x_true = sqrt(x_true.dot(x_true));
    T normA_x_true = sqrt(x_true.dot(A * x_true));
    
    CGResult<T> result;
    result.hist_relres_2.reserve(max_iter + 1);
    result.hist_relerr_2.reserve(max_iter + 1);
    result.hist_relerr_A.reserve(max_iter + 1);
    
    // Initialize residual: r = b - Ax
    VectorType r = b - A * x;
    
    // Calculate initial error vector
    VectorType err = x_true - x;
    
    // Store initial convergence metrics
    T initial_residual_norm = sqrt(r.dot(r));
    result.initial_residual_norm = to_double(initial_residual_norm);
    result.hist_relres_2.push_back(to_double(initial_residual_norm / norm2_b));
    result.hist_relerr_2.push_back(to_double(sqrt(err.dot(err)) / norm2_x_true));
    result.hist_relerr_A.push_back(to_double(sqrt(err.dot(A * err)) / normA_x_true));
    
    // Initialize search direction p = r
    VectorType p = r;
    
    // Initialize rho for beta calculation
    T rho_old = r.dot(r);
    
    // Main CG iteration loop
    bool is_converged = false;
    int iter_final = 0;
    
    for (int iter = 1; iter <= max_iter; ++iter) {
        // Compute matrix-vector product
        VectorType w = A * p;
        
        // Compute denominator for step size
        T sigma = p.dot(w);
        
        // Compute step size α = (r,r) / (p,Ap)
        T alpha = rho_old / sigma;
        
        // Update solution: x = x + α*p
        x = x + alpha * p;
        
        // Update residual: r = r - α*Ap
        r = r - alpha * w;
        
        // Compute current error for analysis
        err = x_true - x;
        
        // Record convergence metrics
        result.hist_relres_2.push_back(to_double(sqrt(r.dot(r)) / norm2_b));
        result.hist_relerr_2.push_back(to_double(sqrt(err.dot(err)) / norm2_x_true));
        result.hist_relerr_A.push_back(to_double(sqrt(err.dot(A * err)) / normA_x_true));
        
        // Check convergence: ||r||₂ / ||b||₂ < tolerance
        if (result.hist_relres_2.back() < tolerance) {
            is_converged = true;
            iter_final = iter;
            break;
        }
        
        // Compute new inner product for next β
        T rho_new = r.dot(r);
        
        // Compute β = (r_{k+1},r_{k+1}) / (r_k,r_k)
        T beta = rho_new / rho_old;
        
        // Update for next iteration
        rho_old = rho_new;
        
        // Update search direction: p = r + β*p
        p = r + beta * p;
        
        iter_final = iter;
    }
    
    // Finalize results
    result.iterations_performed = iter_final;
    result.converged = is_converged;
    result.final_residual_norm = result.hist_relres_2.back();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    result.computation_time = duration.count() / 1000.0;
    
    // Compute true residual to check for gap with computed residual
    VectorType true_residual = b - A * x;
    T true_residual_norm = sqrt(true_residual.dot(true_residual));
    result.true_relres_2 = to_double(true_residual_norm / norm2_b);
    
    return result;
}

/// Print formatted results from CG solver
/// 
/// @param result CG solver results
/// @param problem_name Optional problem identifier for display
template<typename T>
void print_results(const CGResult<T>& result, const std::string& problem_name = "") {
    std::cout << "========================== " << std::endl;
    std::cout << "Numerical Results. " << std::endl;
    if (!problem_name.empty()) {
        std::cout << "Problem: " << problem_name << " " << std::endl;
    }
    std::cout << "Precision: " << result.precision_name << " (" 
              << bailey::PrecisionTraits<T>::decimal_digits() << " digits)" << std::endl;
    std::cout << "========================== " << std::endl;

    if (result.converged) {
        std::cout << "Converged! (iter = " << result.iterations_performed << ")" << std::endl;
    } else {
        std::cout << "NOT converged. (max_iter = " << result.iterations_performed << ")" << std::endl;
    }

    std::cout << "# Iter.: " << result.iterations_performed << std::endl;
    std::cout << std::fixed << std::setprecision(3) << "Time[s]: " << result.computation_time << std::endl;
    std::cout << std::scientific << std::setprecision(2);
    
    // Display final convergence metrics
    int final_idx = result.iterations_performed;
    std::cout << "Relres_2norm = " << result.hist_relres_2[final_idx] << std::endl;
    std::cout << "True_Relres_2norm = " << result.true_relres_2 << std::endl;
    std::cout << "Relerr_2norm = " << result.hist_relerr_2[final_idx] << std::endl;
    std::cout << "Relerr_Anorm = " << result.hist_relerr_A[final_idx] << std::endl;
    std::cout << "========================== " << std::endl;
    std::cout << std::endl;
}

/// Resolve maximum iteration count from user specification
/// 
/// @param spec Either absolute count (int) or coefficient (double)
/// @param matrix_size Size of the matrix for coefficient multiplication
/// @return Absolute maximum iteration count
inline int resolve_max_iterations(const std::variant<int, double>& spec, int matrix_size) {
    if (std::holds_alternative<int>(spec)) {
        return std::get<int>(spec);
    } else {
        return static_cast<int>(std::get<double>(spec) * matrix_size);
    }
}

} // namespace algorithms