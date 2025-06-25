tic;
A = [4 -1 -1 0;
     -1 4 0 -1;
     -1 0 4 -1;
     0 -1 -1 4];

% [L, D, deltaA] = incompleteCholeskyDecomposition(A);
L = ichol(sparse(A)); % LL' format
d = diag(L); % diagonal vector
d2 = d .^ 2;
L = L .* (1 ./ d'); % unit diagonalization
deltaA = A - L .* d2' * L';
toc;

fprintf('L:\n');
disp(full(L));
fprintf('D:\n');
disp(full(d2));
fprintf('deltaA:\n');
disp(deltaA)
