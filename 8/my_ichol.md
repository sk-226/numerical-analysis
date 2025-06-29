# my_ichol_*.m について

- 対角ベクトルのような要素が連続しているベクトルについては, spdiags や sparse を使うよりもメモリ連続なベクトル形式でしたほうが連続してアクセス, 計算できるので早い.
  - 一方で, 非ゼロ要素が連続せず, 散在している場合 (疎行列) は CRS などの圧縮形式を使うほうがよい.

メモリの取り方や BLAS での実装イメージを持つといいかも！！

```matlab
% 何回か実行
% RESULT:
% my_ichol_vec2 (使いまわしありなら) >= my_ichol_vec >>>> my_ichol_spdiags

>> my_ichol_spdiags
Elapsed time is 0.000520 seconds.

>> my_ichol_vec
Elapsed time is 0.000138 seconds.

>> my_ichol_vec2
Elapsed time is 0.000144 seconds.
```
