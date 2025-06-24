A = [4 -1 -1 0;
     -1 4 0 -1;
     -1 0 4 -1;
     0 -1 -1 4];

% [L, D, deltaA] = incompleteCholeskyDecomposition(A);
L = ichol(sparse(A)); % LL' format
d = diag(L) .^ 2; % diagonal vector
D = spdiags(d, 0, size(A, 1), size(A, 2)); % make D a sparse main-diagonal matrix
L = L * spdiags(1 ./ diag(L), 0, size(A, 1), size(A, 2)); % unit diagonalization
deltaA = A - L * D * L';

fprintf('L:\n');
disp(full(L));
fprintf('D:\n');
disp(full(D));
fprintf('deltaA:\n');
disp(deltaA)
