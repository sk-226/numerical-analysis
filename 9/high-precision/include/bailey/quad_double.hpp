#pragma once

#include <iostream>
#include <string>
#include <cmath>
#include <Eigen/Sparse>

// ==============================================================================
//  Bailey QX高精度算術ライブラリとの連携のためのQuadDouble型定義
// ==============================================================================

extern "C" {
    void qxadd_(const double* a, const double* b, double* c);      // c = a + b
    void qxsub_(const double* a, const double* b, double* c);      // c = a - b
    void qxmul_(const double* a, const double* b, double* c);      // c = a * b
    void qxdiv_(const double* a, const double* b, double* c);      // c = a / b
    void qxdqd_(const double* d, double* a);                       // a = (double)d
    void qxsqrt_(const double* a, double* b);                      // b = sqrt(a)
    void qxtoqd_(const double* a, int* n, char* c, int cl);
}

struct QuadDouble {
    double qd[4] = {0.0, 0.0, 0.0, 0.0};
    
    QuadDouble() = default;
    QuadDouble(double val) { qxdqd_(&val, qd); }
};

// --- Basic Arithmetic Operators ---
inline QuadDouble operator+(const QuadDouble& a, const QuadDouble& b) { 
    QuadDouble r; qxadd_(a.qd, b.qd, r.qd); return r; 
}

inline QuadDouble operator-(const QuadDouble& a, const QuadDouble& b) { 
    QuadDouble r; qxsub_(a.qd, b.qd, r.qd); return r; 
}

inline QuadDouble operator*(const QuadDouble& a, const QuadDouble& b) { 
    QuadDouble r; qxmul_(a.qd, b.qd, r.qd); return r; 
}

inline QuadDouble operator/(const QuadDouble& a, const QuadDouble& b) { 
    QuadDouble r; qxdiv_(a.qd, b.qd, r.qd); return r; 
}

// --- Assignment Operators ---
inline QuadDouble& operator+=(QuadDouble& a, const QuadDouble& b) { 
    a = a + b; return a; 
}

inline QuadDouble& operator-=(QuadDouble& a, const QuadDouble& b) { 
    a = a - b; return a; 
}

inline QuadDouble& operator*=(QuadDouble& a, const QuadDouble& b) { 
    a = a * b; return a; 
}

inline QuadDouble& operator/=(QuadDouble& a, const QuadDouble& b) { 
    a = a / b; return a; 
}

// --- Mathematical Functions ---
inline QuadDouble sqrt(const QuadDouble& a) { 
    QuadDouble r; qxsqrt_(a.qd, r.qd); return r; 
}

// --- Type Conversion ---
inline double to_double(const QuadDouble& a) { 
    char s[70]; 
    int d = 50; // Use higher precision instead of 15
    qxtoqd_(a.qd, &d, s, sizeof(s));
    
    std::string str(s, sizeof(s));
    size_t end = str.find('\0');
    if (end != std::string::npos) {
        str = str.substr(0, end);
    }
    
    try {
        return std::stod(str);
    } catch (...) {
        return a.qd[0];
    }
}

// --- Stream Output ---
inline std::ostream& operator<<(std::ostream& os, const QuadDouble& q) {
    char s[70]; 
    int d = 50; // Use higher precision for output
    qxtoqd_(q.qd, &d, s, sizeof(s)); 
    os << std::string(s, sizeof(s)); 
    return os;
}

// --- Eigen Integration ---
namespace Eigen {
    template<> struct NumTraits<QuadDouble> : GenericNumTraits<QuadDouble> {
        typedef QuadDouble Real; 
        typedef QuadDouble NonInteger; 
        typedef QuadDouble Nested;
        enum { 
            IsComplex = 0, 
            IsInteger = 0, 
            IsSigned = 1, 
            RequireInitialization = 1, 
            ReadCost = 4, 
            AddCost = 32, 
            MulCost = 64 
        };
    };
}

// --- Type Aliases ---
using SpMat_QD = Eigen::SparseMatrix<QuadDouble>;
using Vec_QD = Eigen::Vector<QuadDouble, Eigen::Dynamic>;