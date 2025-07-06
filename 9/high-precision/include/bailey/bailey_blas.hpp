#pragma once

#include "precision_traits.hpp"
#include <vector>

namespace bailey_blas {

/// Template-based BLAS interface for unified high-performance operations
/// Provides BLAS-like operations for all precision types including DD/DQ/QX
template<typename T>
struct BLASTraits {
    
    /// Compute dot product: result = x^T * y
    static T dot(const std::vector<T>& x, const std::vector<T>& y) {
        T result = T(0.0);
        const size_t n = x.size();
        
        // Optimized loop with reduced function call overhead
        for (size_t i = 0; i < n; ++i) {
            result = result + x[i] * y[i];
        }
        return result;
    }
    
    /// AXPY operation: y = alpha * x + y
    static void axpy(const T& alpha, const std::vector<T>& x, std::vector<T>& y) {
        const size_t n = x.size();
        
        // Cache-friendly sequential access
        for (size_t i = 0; i < n; ++i) {
            y[i] = y[i] + alpha * x[i];
        }
    }
    
    /// Scalar multiplication: x = alpha * x
    static void scal(const T& alpha, std::vector<T>& x) {
        const size_t n = x.size();
        
        for (size_t i = 0; i < n; ++i) {
            x[i] = alpha * x[i];
        }
    }
    
    /// Vector copy: y = x
    static void copy(const std::vector<T>& x, std::vector<T>& y) {
        y = x; // STL optimized copy
    }
    
    /// Vector norm: ||x||_2
    static T nrm2(const std::vector<T>& x) {
        return sqrt(dot(x, x));
    }
};

/// Specialization for double precision - use native BLAS
template<>
struct BLASTraits<double> {
    
    static double dot(const std::vector<double>& x, const std::vector<double>& y) {
        // Use cblas_ddot for maximum performance
        return cblas_ddot(static_cast<int>(x.size()), x.data(), 1, y.data(), 1);
    }
    
    static void axpy(const double& alpha, const std::vector<double>& x, std::vector<double>& y) {
        cblas_daxpy(static_cast<int>(x.size()), alpha, x.data(), 1, y.data(), 1);
    }
    
    static void scal(const double& alpha, std::vector<double>& x) {
        cblas_dscal(static_cast<int>(x.size()), alpha, x.data(), 1);
    }
    
    static void copy(const std::vector<double>& x, std::vector<double>& y) {
        cblas_dcopy(static_cast<int>(x.size()), x.data(), 1, y.data(), 1);
    }
    
    static double nrm2(const std::vector<double>& x) {
        return cblas_dnrm2(static_cast<int>(x.size()), x.data(), 1);
    }
};

/// Convenience functions for unified interface
template<typename T>
T dot(const std::vector<T>& x, const std::vector<T>& y) {
    return BLASTraits<T>::dot(x, y);
}

template<typename T>
void axpy(const T& alpha, const std::vector<T>& x, std::vector<T>& y) {
    BLASTraits<T>::axpy(alpha, x, y);
}

template<typename T>
void scal(const T& alpha, std::vector<T>& x) {
    BLASTraits<T>::scal(alpha, x);
}

template<typename T>
void copy(const std::vector<T>& x, std::vector<T>& y) {
    BLASTraits<T>::copy(x, y);
}

template<typename T>
T nrm2(const std::vector<T>& x) {
    return BLASTraits<T>::nrm2(x);
}

} // namespace bailey_blas