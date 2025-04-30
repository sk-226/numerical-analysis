function newton_secant_demo
  %  f(x)  と  f'(x)
  f  = @(x) x^2 - 7*x + 12;  % f(x) = x^2 - 7x + 12
  df = @(x) 2*x - 7;          % f'(x) = 2x - 7

  alpha   = 3;          % true root
  eps_res = 1e-9;       % residual threshold for convergence
  maxIter = 100;        % maximum number of iterations

  % ---------- Newton method ----------
  xn  = 1;                  % x_0
  errN(1) = abs(xn - alpha);
  for k = 1:maxIter
    if abs(f(xn)) < eps_res, break; end
    xn = xn - f(xn)/df(xn);
    errN(k+1) = abs(xn - alpha);
  end

  % ---------- Secant method ----------
  x0 = 0.5;  x1 = 1;        % x_0, x_1
  errS = [abs(x0 - alpha)  abs(x1 - alpha)];
  for k = 2:maxIter
    if abs(f(x1)) < eps_res, break; end
    x2 = x1 - f(x1)*(x1 - x0)/(f(x1) - f(x0));
    x0 = x1;  x1 = x2;
    errS(k+1) = abs(x1 - alpha);
  end

  % ---------- plot error history ----------
  figure
  semilogy(0:numel(errN)-1, errN,  'o-', 'DisplayName','Newton'); hold on
  semilogy(0:numel(errS)-1, errS,  's-', 'DisplayName','Secant');

  % ---- highlight the last three iterations for each method ----
  idxN = max(1, numel(errN)-2):numel(errN);   % indices of final 3 Newton points
  idxS = max(1, numel(errS)-2):numel(errS);   % indices of final 3 Secant points

  % Plot larger filled markers on those points
  semilogy(idxN-1, errN(idxN), 'o','MarkerFaceColor', 'k', 'HandleVisibility', 'off');
  semilogy(idxS-1, errS(idxS), 's','MarkerFaceColor', 'k', 'HandleVisibility', 'off');

  % Annotate each highlighted point with its error value
  for i = idxN
    text(i-1, errN(i), sprintf('%.1e', errN(i)), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
  end
  for i = idxS
    text(i-1, errS(i), sprintf('%.1e', errS(i)), 'VerticalAlignment', 'top', 'HorizontalAlignment', 'left');
  end

  xlabel('Iteration  k'), ylabel('|\epsilon_k| = |x_k - 3| (log_{10})');
  title('Convergence history  for  f(x)=x^2-7x+12');
  legend show, grid on

  set(gca, 'FontSize', 14);

  saveas(gcf, 'newton_secant_demo.svg'); % save figure
end
