/**
 * @file quad_test.c
 * @author Suguru Kurita
 * @brief Check the precision of __float128. compile with -lquadmath
 * @details
 * Compile `gcc -std=c11 -o quad_test quad_test.c -lquadmath `
 * @date 2025-04-16
 *
 */
#include <quadmath.h>
#include <stdio.h>

int main(void) {
  __float128 x = 1.0Q / 3.0Q;
  char buf[128];

  quadmath_snprintf(buf, sizeof(buf), "%.36Qg", x);
  printf("__float128: %s\n", buf);

  printf("FLT128_DIG: %d\n", FLT128_DIG);

  return 0;
}
