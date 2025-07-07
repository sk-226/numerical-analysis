#pragma once

#include <iostream>
#include <string>
#include <cmath>
#include <Eigen/Sparse>

// DD (Double-Double) precision arithmetic using Bailey's DDFUN library
extern "C" {
    void ddadd_(const double* a, const double* b, double* c);      // c = a + b
    void ddsub_(const double* a, const double* b, double* c);      // c = a - b
    void ddmul_(const double* a, const double* b, double* c);      // c = a * b
    void dddiv_(const double* a, const double* b, double* c);      // c = a / b
    void dddqd_(const double* d, double* a);                       // a = (double)d
    void ddsqrt_(const double* a, double* b);                      // b = sqrt(a)
    void ddtoqd_(const double* a, int* n, char* c, int cl);
}

namespace bailey {

struct DDNumber {
    double dd[2] = {0.0, 0.0};
    
    DDNumber() = default;
    DDNumber(double val) { dddqd_(&val, dd); }
};

// Basic Arithmetic Operators
inline DDNumber operator+(const DDNumber& a, const DDNumber& b) { 
    DDNumber r; ddadd_(a.dd, b.dd, r.dd); return r; 
}

inline DDNumber operator-(const DDNumber& a, const DDNumber& b) { 
    DDNumber r; ddsub_(a.dd, b.dd, r.dd); return r; 
}

inline DDNumber operator*(const DDNumber& a, const DDNumber& b) { 
    DDNumber r; ddmul_(a.dd, b.dd, r.dd); return r; 
}

inline DDNumber operator/(const DDNumber& a, const DDNumber& b) { 
    DDNumber r; dddiv_(a.dd, b.dd, r.dd); return r; 
}

// Assignment Operators
inline DDNumber& operator+=(DDNumber& a, const DDNumber& b) { 
    a = a + b; return a; 
}

inline DDNumber& operator-=(DDNumber& a, const DDNumber& b) { 
    a = a - b; return a; 
}

inline DDNumber& operator*=(DDNumber& a, const DDNumber& b) { 
    a = a * b; return a; 
}

inline DDNumber& operator/=(DDNumber& a, const DDNumber& b) { 
    a = a / b; return a; 
}

// Mathematical Functions
inline DDNumber sqrt(const DDNumber& a) { 
    DDNumber r; ddsqrt_(a.dd, r.dd); return r; 
}

// Type Conversion
inline double to_double(const DDNumber& a) { 
    char s[50]; 
    int d = 32; // Use full DD precision instead of 15
    ddtoqd_(a.dd, &d, s, sizeof(s));
    
    std::string str(s, sizeof(s));
    size_t end = str.find('\0');
    if (end != std::string::npos) {
        str = str.substr(0, end);
    }
    
    try {
        return std::stod(str);
    } catch (...) {
        return a.dd[0];
    }
}

// Stream Output
inline std::ostream& operator<<(std::ostream& os, const DDNumber& d) {
    char s[50]; 
    int digits = 32; // Use full DD precision
    ddtoqd_(d.dd, &digits, s, sizeof(s)); 
    os << std::string(s, sizeof(s)); 
    return os;
}

} // namespace bailey

// Eigen Integration
namespace Eigen {
    template<> struct NumTraits<bailey::DDNumber> : GenericNumTraits<bailey::DDNumber> {
        typedef bailey::DDNumber Real; 
        typedef bailey::DDNumber NonInteger; 
        typedef bailey::DDNumber Nested;
        enum { 
            IsComplex = 0, 
            IsInteger = 0, 
            IsSigned = 1, 
            RequireInitialization = 1, 
            ReadCost = 2, 
            AddCost = 16, 
            MulCost = 32 
        };
    };
}