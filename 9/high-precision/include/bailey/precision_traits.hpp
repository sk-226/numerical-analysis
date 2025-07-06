#pragma once

#include <Eigen/Sparse>
#include <string_view>

// Include precision type definitions to resolve template specialization issues
#include "quad_double.hpp"
#include "dd_arithmetic.hpp"
#include "dq_arithmetic.hpp"

namespace bailey {

/// Template traits system for precision-agnostic numerical computing
/// 
/// Provides unified interface for different arithmetic precision levels,
/// enabling a single algorithm implementation to work across multiple
/// precision types (double, DD, DQ, QX).
template<typename T>
struct PrecisionTraits {
    using scalar_type = T;
    using matrix_type = Eigen::SparseMatrix<T>;
    using vector_type = Eigen::Vector<T, Eigen::Dynamic>;
    
    static constexpr const char* name() { return "Unknown"; }
    static constexpr int decimal_digits() { return 0; }
};

// Template specializations for supported precision types

/// Double-Double precision (DD) - ~32 decimal digits
/// Uses Bailey's DDFUN library for extended precision arithmetic
template<>
struct PrecisionTraits<bailey::DDNumber> {
    using scalar_type = bailey::DDNumber;
    using matrix_type = Eigen::SparseMatrix<bailey::DDNumber>;
    using vector_type = Eigen::Vector<bailey::DDNumber, Eigen::Dynamic>;
    
    static constexpr const char* name() { return "DD"; }
    static constexpr int decimal_digits() { return 32; }
};

/// Quad-Double precision (DQ) - ~64 decimal digits  
/// Uses Bailey's DQFUN library for very high precision arithmetic
template<>
struct PrecisionTraits<bailey::DQNumber> {
    using scalar_type = bailey::DQNumber;
    using matrix_type = Eigen::SparseMatrix<bailey::DQNumber>;
    using vector_type = Eigen::Vector<bailey::DQNumber, Eigen::Dynamic>;
    
    static constexpr const char* name() { return "DQ"; }
    static constexpr int decimal_digits() { return 64; }
};

// Type aliases for convenience  
using DDTraits = PrecisionTraits<bailey::DDNumber>;
using DQTraits = PrecisionTraits<bailey::DQNumber>;

} // namespace bailey

/// Extended Quad precision (QX) - ~128 decimal digits
/// Uses Bailey's QXFUN library for maximum precision arithmetic
/// Note: QuadDouble is in global namespace for legacy compatibility
template<>
struct bailey::PrecisionTraits<QuadDouble> {
    using scalar_type = QuadDouble;
    using matrix_type = Eigen::SparseMatrix<QuadDouble>;
    using vector_type = Eigen::Vector<QuadDouble, Eigen::Dynamic>;
    
    static constexpr const char* name() { return "QX"; }
    static constexpr int decimal_digits() { return 128; }
};

/// Standard IEEE 754 double precision - ~15 decimal digits
/// Native C++ floating-point arithmetic for performance baseline
template<>
struct bailey::PrecisionTraits<double> {
    using scalar_type = double;
    using matrix_type = Eigen::SparseMatrix<double>;
    using vector_type = Eigen::Vector<double, Eigen::Dynamic>;
    
    static constexpr const char* name() { return "Double"; }
    static constexpr int decimal_digits() { return 15; }
};

namespace bailey {
using QXTraits = PrecisionTraits<QuadDouble>;
using DoubleTraits = PrecisionTraits<double>;
} // namespace bailey