%% Test script for Cholesky decomposition with Hilbert matrices
%
%  This script tests the numerical stability of Cholesky decomposition and
%  modified Cholesky decomposition using Hilbert matrices of different sizes.
%  Hilbert matrices are known to be ill-conditioned and provide a good test
%  for numerical algorithms.

clear; clc;

fprintf('=== Cholesky Decomposition Test with Hilbert Matrices ===\n\n');

% Test dimensions
test_sizes = [5, 10, 15];

for idx = 1:length(test_sizes)
    n = test_sizes(idx);

    fprintf('Test %d: Hilbert matrix of size %d×%d\n', idx, n, n);
    fprintf('========================================\n');

    % Generate Hilbert matrix
    H = hilb(n);
    fprintf('Generated %d×%d Hilbert matrix\n', n, n);

    % Display condition number
    cond_H = cond(H);
    fprintf('Condition number: %.2e\n', cond_H);

    % Display matrix properties
    fprintf('Matrix properties:\n');
    fprintf('  - Symmetric: %s\n', mat2str(issymmetric(H)));
    fprintf('  - Positive definite: %s\n', mat2str(all(eig(H) > 0)));
    fprintf('  - Smallest eigenvalue: %.2e\n', min(eig(H)));
    fprintf('  - Largest eigenvalue: %.2e\n', max(eig(H)));

    fprintf('\n--- Standard Cholesky Decomposition ---\n');

    % Test standard Cholesky decomposition
    try
        tic;
        L_chol = choleskyDecomposition(H);
        time_chol = toc;

        % Verify reconstruction
        H_reconstructed = L_chol * L_chol';
        error_chol = norm(H_reconstructed - H, 'fro');
        relative_error_chol = error_chol / norm(H, 'fro');

        fprintf('SUCCESS: Standard Cholesky decomposition completed\n');
        fprintf('  - Computation time: %.4f seconds\n', time_chol);
        fprintf('  - Reconstruction error (Frobenius): %.2e\n', error_chol);
        fprintf('  - Relative error: %.2e\n', relative_error_chol);

        % Check if error is reasonable
        if relative_error_chol < 1e-10
            fprintf('  - Status: EXCELLENT accuracy\n');
        elseif relative_error_chol < 1e-6
            fprintf('  - Status: GOOD accuracy\n');
        elseif relative_error_chol < 1e-3
            fprintf('  - Status: ACCEPTABLE accuracy\n');
        else
            fprintf('  - Status: POOR accuracy - numerical issues likely\n');
        end

    catch ME
        fprintf('FAILED: Standard Cholesky decomposition failed\n');
        fprintf('  - Error: %s\n', ME.message);
        L_chol = [];
        time_chol = NaN;
        error_chol = NaN;
        relative_error_chol = NaN;
    end

    fprintf('\n--- Modified Cholesky Decomposition ---\n');

    % Test modified Cholesky decomposition
    try
        tic;
        [L_tilde, D] = modifiedCholeskyDecomposition(H);
        time_mod = toc;

        % Verify reconstruction
        H_reconstructed_mod = L_tilde * D * L_tilde';
        error_mod = norm(H_reconstructed_mod - H, 'fro');
        relative_error_mod = error_mod / norm(H, 'fro');

        fprintf('SUCCESS: Modified Cholesky decomposition completed\n');
        fprintf('  - Computation time: %.4f seconds\n', time_mod);
        fprintf('  - Reconstruction error (Frobenius): %.2e\n', error_mod);
        fprintf('  - Relative error: %.2e\n', relative_error_mod);

        % Check diagonal matrix condition
        min_diag = min(diag(D));
        fprintf('  - Smallest diagonal element in D: %.2e\n', min_diag);

        % Check if error is reasonable
        if relative_error_mod < 1e-10
            fprintf('  - Status: EXCELLENT accuracy\n');
        elseif relative_error_mod < 1e-6
            fprintf('  - Status: GOOD accuracy\n');
        elseif relative_error_mod < 1e-3
            fprintf('  - Status: ACCEPTABLE accuracy\n');
        else
            fprintf('  - Status: POOR accuracy - numerical issues likely\n');
        end

    catch ME
        fprintf('FAILED: Modified Cholesky decomposition failed\n');
        fprintf('  - Error: %s\n', ME.message);
        L_tilde = [];
        D = [];
        time_mod = NaN;
        error_mod = NaN;
        relative_error_mod = NaN;
    end

    fprintf('\n--- Comparison with MATLAB built-in ---\n');

    % Compare with MATLAB's chol function
    try
        tic;
        L_matlab = chol(H, 'lower');
        time_matlab = toc;

        H_matlab_reconstructed = L_matlab * L_matlab';
        error_matlab = norm(H_matlab_reconstructed - H, 'fro');
        relative_error_matlab = error_matlab / norm(H, 'fro');

        fprintf('MATLAB chol function:\n');
        fprintf('  - Computation time: %.4f seconds\n', time_matlab);
        fprintf('  - Reconstruction error (Frobenius): %.2e\n', error_matlab);
        fprintf('  - Relative error: %.2e\n', relative_error_matlab);

    catch ME
        fprintf('MATLAB chol function FAILED: %s\n', ME.message);
        time_matlab = NaN;
        error_matlab = NaN;
        relative_error_matlab = NaN;
    end

    fprintf('\n--- Performance Summary ---\n');

    if ~isnan(time_chol) && ~isnan(time_mod)

        if time_chol < time_mod
            fprintf('Standard Cholesky is %.2fx faster than modified\n', time_mod / time_chol);
        else
            fprintf('Modified Cholesky is %.2fx faster than standard\n', time_chol / time_mod);
        end

    end

    if ~isnan(relative_error_chol) && ~isnan(relative_error_mod)

        if relative_error_chol < relative_error_mod
            fprintf('Standard Cholesky is more accurate by factor %.2e\n', relative_error_mod / relative_error_chol);
        else
            fprintf('Modified Cholesky is more accurate by factor %.2e\n', relative_error_chol / relative_error_mod);
        end

    end

    fprintf('\n');
    fprintf('================================================\n\n');
end
