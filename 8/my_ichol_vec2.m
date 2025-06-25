tic;
% --- Incomplete Cholesky decomposition (vector-based implementation) ---
A = [ 4 -1 -1  0;
     -1  4  0 -1;
     -1  0  4 -1;
      0 -1 -1  4];

% 1) Standard (LL') incomplete Cholesky factor
L = ichol(sparse(A));          % lower-triangular, non-unit diagonal

% 2) Extract diagonal once and reuse it as a *vector* throughout
d  = diag(L);                  % length-n column vector
d2 = d.^2;                     % squared diagonal (corresponds to D’s entries)

% 3) Unit-diagonalise L in-place ─ scale each column j by 1/d(j)
L  = L .* (1./d');             % uses implicit expansion, no spdiags/diag

% 4) Form L*D without constructing D:
LD = L .* d2';                 % column-wise scaling (equivalent to L*diag(d2))

% 5) Error matrix ΔA = A − L D Lᵀ
deltaA = A - LD * L';

toc;

% 6) Display results
fprintf('L (unit-lower-triangular):\n');
disp(full(L));

fprintf('d2 (vector holding diag(D)):\n');
disp(d2);                      % D itself is never built—only its diagonal

fprintf('deltaA = A − L*D*Lʹ:\n');
disp(deltaA);

