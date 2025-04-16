function inv_square_sum
  % double型の有効数字は約15桁なので、整数部1桁・小数部14桁で表示する
  n = 10000;
  s1 = 0;
  s2 = 0;

  % 前からの総和
  for i = 1:n
      s1 = s1 + inv_square(i);
  end

  % 後ろからの総和
  for i = n:-1:1
      s2 = s2 + inv_square(i);
  end

  % 結果を指数表記で表示（%.14e 相当）
  fprintf('s1 = %.14e\n', s1);
  fprintf('s2 = %.14e\n', s2);
end

% inv_square 関数：1/(x*x) を計算する
function y = inv_square(x)
  y = 1 / (x * x);
end
