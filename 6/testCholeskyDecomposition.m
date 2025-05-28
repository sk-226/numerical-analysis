%% Test script for Cholesky and modified Cholesky decomposition implementations
%
%
%  This script tests the Cholesky decomposition and modified Cholesky
%  decomposition implementations, verifying their correctness and
%  comparing with MATLAB's built-in functions.

clear; clc;

fprintf('=== Cholesky and Modified Cholesky Decomposition Tests ===\n\n');

%% Test 1: Simple 2×2 symmetric positive-definite matrix
fprintf('Test 1: 2×2 symmetric positive-definite matrix\n');
A1 = [4, 2; 2, 5];
b1 = [6; 7];

fprintf('Matrix A1:\n');
disp(A1);
fprintf('Vector b1:\n');
disp(b1);

% ----------------------- Cholesky decomposition test --------------------
L1 = choleskyDecomposition(A1);
fprintf('Cholesky decomposition L1:\n');
disp(L1);

% verification: check if L1 * L1^T = A1
reconstruction1 = L1 * L1';
fprintf('Reconstruction check L1 * L1^T:\n');
disp(reconstruction1);
fprintf('Frobenius norm error: %.2e\n', norm(reconstruction1 - A1, 'fro'));

% ------------------ modified Cholesky decomposition test ---------------
[L_tilde1, D1] = modifiedCholeskyDecomposition(A1);
fprintf('\nModified Cholesky decomposition L_tilde1:\n');
disp(L_tilde1);
fprintf('Diagonal matrix D1:\n');
disp(D1);

% verification: check if L_tilde1 * D1 * L_tilde1^T = A1
reconstruction1_mod = L_tilde1 * D1 * L_tilde1';
fprintf('Reconstruction check L_tilde1 * D1 * L_tilde1^T:\n');
disp(reconstruction1_mod);
fprintf('Frobenius norm error: %.2e\n', norm(reconstruction1_mod - A1, 'fro'));

% ----------------------- linear system solution test -------------------
x1_chol = choleskyMethod(A1, b1);
x1_mod = modifiedCholeskyMethod(A1, b1);

fprintf('\nSolution by Cholesky method:\n');
disp(x1_chol);
fprintf('Solution by modified Cholesky method:\n');
disp(x1_mod);

% verification of solutions
residual1_chol = norm(A1 * x1_chol - b1);
residual1_mod = norm(A1 * x1_mod - b1);
fprintf('Cholesky method residual: %.2e\n', residual1_chol);
fprintf('Modified Cholesky method residual: %.2e\n', residual1_mod);

fprintf('--------------------------------\n');

%% Test 2: 3×3 symmetric positive-definite matrix
fprintf('\n\nTest 2: 3×3 symmetric positive-definite matrix\n');
A2 = [9, 3, 1; 3, 6, 2; 1, 2, 4];
b2 = [13; 11; 7];

fprintf('Matrix A2:\n');
disp(A2);
fprintf('Vector b2:\n');
disp(b2);

% ----------------------- Cholesky decomposition test --------------------
L2 = choleskyDecomposition(A2);
fprintf('Cholesky decomposition L2:\n');
disp(L2);

% ------------------ modified Cholesky decomposition test ---------------
[L_tilde2, D2] = modifiedCholeskyDecomposition(A2);
fprintf('Modified Cholesky decomposition L_tilde2:\n');
disp(L_tilde2);
fprintf('Diagonal matrix D2:\n');
disp(D2);

% ----------------------- reconstruction verification -------------------
reconstruction2 = L2 * L2';
reconstruction2_mod = L_tilde2 * D2 * L_tilde2';
fprintf('Cholesky reconstruction error: %.2e\n', norm(reconstruction2 - A2, 'fro'));
fprintf('Modified Cholesky reconstruction error: %.2e\n', norm(reconstruction2_mod - A2, 'fro'));

% ----------------------- linear system solution test -------------------
x2_chol = choleskyMethod(A2, b2);
x2_mod = modifiedCholeskyMethod(A2, b2);

fprintf('\nSolution by Cholesky method:\n');
disp(x2_chol);
fprintf('Solution by modified Cholesky method:\n');
disp(x2_mod);

% verification of solutions
residual2_chol = norm(A2 * x2_chol - b2);
residual2_mod = norm(A2 * x2_mod - b2);
fprintf('Cholesky method residual: %.2e\n', residual2_chol);
fprintf('Modified Cholesky method residual: %.2e\n', residual2_mod);

fprintf('--------------------------------\n');

%% Comparison with MATLAB built-in functions
fprintf('\n\nComparison with MATLAB built-in functions:\n');

% MATLAB's chol function
L_matlab = chol(A2, 'lower');
fprintf('MATLAB chol function (lower triangular):\n');
disp(L_matlab);
fprintf('Reconstruction error: %.2e\n', norm(L_matlab * L_matlab' - A2, 'fro'));

% MATLAB's mldivide (backslash) operator
x_matlab = A2 \ b2;
fprintf('MATLAB backslash operator solution:\n');
disp(x_matlab);
fprintf('Residual: %.2e\n', norm(A2 * x_matlab - b2));

fprintf('\n=== All tests completed ===\n');
