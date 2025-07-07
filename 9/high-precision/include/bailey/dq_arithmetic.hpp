#pragma once

#include <iostream>
#include <string>
#include <cmath>
#include <Eigen/Sparse>

// DQ (Quad-Double) precision arithmetic using Bailey's DQFUN library
extern "C" {
    void dqadd_(const double* a, const double* b, double* c);      // c = a + b
    void dqsub_(const double* a, const double* b, double* c);      // c = a - b
    void dqmul_(const double* a, const double* b, double* c);      // c = a * b
    void dqdiv_(const double* a, const double* b, double* c);      // c = a / b
    void dqdqd_(const double* d, double* a);                       // a = (double)d
    void dqsqrt_(const double* a, double* b);                      // b = sqrt(a)
    void dqtoqd_(const double* a, int* n, char* c, int cl);
}

namespace bailey {

struct DQNumber {
    double dq[4] = {0.0, 0.0, 0.0, 0.0};
    
    DQNumber() = default;
    DQNumber(double val) { dqdqd_(&val, dq); }
};

// Basic Arithmetic Operators
inline DQNumber operator+(const DQNumber& a, const DQNumber& b) { 
    DQNumber r; dqadd_(a.dq, b.dq, r.dq); return r; 
}

inline DQNumber operator-(const DQNumber& a, const DQNumber& b) { 
    DQNumber r; dqsub_(a.dq, b.dq, r.dq); return r; 
}

inline DQNumber operator*(const DQNumber& a, const DQNumber& b) { 
    DQNumber r; dqmul_(a.dq, b.dq, r.dq); return r; 
}

inline DQNumber operator/(const DQNumber& a, const DQNumber& b) { 
    DQNumber r; dqdiv_(a.dq, b.dq, r.dq); return r; 
}

// Assignment Operators
inline DQNumber& operator+=(DQNumber& a, const DQNumber& b) { 
    a = a + b; return a; 
}

inline DQNumber& operator-=(DQNumber& a, const DQNumber& b) { 
    a = a - b; return a; 
}

inline DQNumber& operator*=(DQNumber& a, const DQNumber& b) { 
    a = a * b; return a; 
}

inline DQNumber& operator/=(DQNumber& a, const DQNumber& b) { 
    a = a / b; return a; 
}

// Mathematical Functions
inline DQNumber sqrt(const DQNumber& a) { 
    DQNumber r; dqsqrt_(a.dq, r.dq); return r; 
}

// Type Conversion
inline double to_double(const DQNumber& a) { 
    char s[70]; 
    int d = 64; // Use full DQ precision instead of 15
    dqtoqd_(a.dq, &d, s, sizeof(s));
    
    std::string str(s, sizeof(s));
    size_t end = str.find('\0');
    if (end != std::string::npos) {
        str = str.substr(0, end);
    }
    
    try {
        return std::stod(str);
    } catch (...) {
        return a.dq[0];
    }
}

// Stream Output
inline std::ostream& operator<<(std::ostream& os, const DQNumber& dq) {
    char s[70]; 
    int digits = 64; // Use full DQ precision
    dqtoqd_(dq.dq, &digits, s, sizeof(s)); 
    os << std::string(s, sizeof(s)); 
    return os;
}

} // namespace bailey

// Eigen Integration
namespace Eigen {
    template<> struct NumTraits<bailey::DQNumber> : GenericNumTraits<bailey::DQNumber> {
        typedef bailey::DQNumber Real; 
        typedef bailey::DQNumber NonInteger; 
        typedef bailey::DQNumber Nested;
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