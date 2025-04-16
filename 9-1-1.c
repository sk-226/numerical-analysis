/**
 * @file 9-1-1.c
 * @author Suguru Kurita
 * @brief This is a test program for question 9-1-1.
 * @details
 * This program compares the sum of 1/(i^2) from 1 to n and from n to 1. 
 * Check this textbook for the details. (小池 直之 他 著, 理工系の基礎 数学 Ⅱ pp.129~134, 丸善出版, 2018.)
 * @date 2025-04-16
 * 
 */
#include <stdio.h>

double inv_square(double x);

int main(void) {
  double s1 = 0;
  double s2 = 0;

  int n = 10000;

  for (int i = 1; i <= n; i++) {
    s1 += inv_square((double)i);
  }

  // 小さい値から足していくほうがgood
  for (int i = n; i >= 1; i--) {
    s2 += inv_square((double)i);
  }

  // check_significant_digits.cからdoubleの有効数字は15桁
  // そのため、整数部1桁, 小数部分を14桁表示
  printf("s1 = %.14e\n", s1);
  printf("s2 = %.14e\n", s2);

  return 0;
}

double inv_square(double x) { return (double)1 / (x * x); }
