/**
 * @file check_significant_digits.c
 * @author Suguru Kurita
 * @brief Check the precision of double and long double
 * @date 2025-04-16
 * 
 */
#include <stdio.h>
#include <float.h>

int main() {
    printf("DBL_DIG = %d\n", DBL_DIG);
    printf("LDBL_DIG = %d\n", LDBL_DIG);
    return 0;
}
