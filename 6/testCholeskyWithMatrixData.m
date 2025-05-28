%% Test Cholesky decomposition accuracy with matrix data files
%
%  This script tests both standard and modified Cholesky decomposition
%  using matrix data from .mat files in the inputs directory.
%  It computes solution accuracy and residuals when the true solution
%  is a vector of all ones.

clear; clc;

fprintf('=== Cholesky Decomposition Accuracy Test with Matrix Data ===\n\n');

% Get the directory of this script and construct path to inputs
script_dir = fileparts(mfilename('fullpath'));
data_dir = fullfile(script_dir, 'inputs');
fprintf('Looking for matrix files in: %s\n', data_dir);

mat_files = {'nos2.mat', 'nos4.mat', 'nos6.mat', 'nos7.mat'};

% Check if inputs directory exists
if ~exist(data_dir, 'dir')
    fprintf('ERROR: Inputs directory not found: %s\n', data_dir);
    return;
end

% Results storage
results = struct();
test_count = 0;

for file_idx = 1:length(mat_files)
    filename = mat_files{file_idx};
    filepath = fullfile(data_dir, filename);

    fprintf('================================================\n');
    fprintf('Testing file: %s\n', filename);
    fprintf('Full path: %s\n', filepath);
    fprintf('================================================\n');

    % Check if file exists
    if ~exist(filepath, 'file')
        fprintf('ERROR: File %s not found\n', filepath);
        fprintf('Available files in directory:\n');
        dir_contents = dir(fullfile(data_dir, '*.mat'));

        for i = 1:length(dir_contents)
            fprintf('  - %s\n', dir_contents(i).name);
        end

        fprintf('\n');
        continue;
    end

    try
        % Load matrix data
        data = load(filepath);

        % Display available variables
        var_names = fieldnames(data);
        fprintf('Variables in file: %s\n', strjoin(var_names, ', '));

        % Try to extract matrix A
        A = [];

        if isfield(data, 'Problem') && isfield(data.Problem, 'A')
            A = data.Problem.A;
            fprintf('Matrix found in Problem.A\n');
        elseif isfield(data, 'A')
            A = data.A;
            fprintf('Matrix found in A\n');
        else
            % Try the first variable that looks like a matrix
            for i = 1:length(var_names)
                var = data.(var_names{i});

                if isnumeric(var) && ismatrix(var) && size(var, 1) == size(var, 2)
                    A = var;
                    fprintf('Matrix found in variable: %s\n', var_names{i});
                    break;
                end

            end

        end

        if isempty(A)
            fprintf('ERROR: No suitable matrix found in %s\n\n', filename);
            continue;
        end

        % Convert to full matrix if sparse
        if issparse(A)
            fprintf('Converting sparse matrix to full matrix\n');
            A = full(A);
        end

        % Display matrix properties
        [n, m] = size(A);
        fprintf('\nMatrix properties:\n');
        fprintf('  - Size: %d × %d\n', n, m);
        fprintf('  - Data type: %s\n', class(A));
        fprintf('  - Is symmetric: %s\n', mat2str(issymmetric(A)));
        fprintf('  - Condition number: %.2e\n', cond(A));

        % Check if matrix is square
        if n ~= m
            fprintf('ERROR: Matrix is not square (%d × %d)\n\n', n, m);
            continue;
        end

        % Make matrix symmetric if it's not
        if ~issymmetric(A)
            fprintf('Making matrix symmetric: A = (A + A'')/2\n');
            A = (A + A') / 2;
        end

        % Check if matrix is positive definite
        eig_vals = eig(A);
        min_eig = min(eig_vals);
        max_eig = max(eig_vals);

        fprintf('  - Smallest eigenvalue: %.2e\n', min_eig);
        fprintf('  - Largest eigenvalue: %.2e\n', max_eig);
        fprintf('  - Is positive definite: %s\n', mat2str(min_eig > 0));

        if min_eig <= 0
            % Make matrix positive definite by adding identity
            shift = abs(min_eig) +1e-6;
            A = A + shift * eye(n);
            fprintf('Made positive definite by adding %.2e * I\n', shift);
            fprintf('  - New smallest eigenvalue: %.2e\n', min(eig(A)));
        end

        % Skip if matrix is too large for demonstration
        if n > 1000
            fprintf('WARNING: Matrix size (%d) is large. Skipping for performance.\n\n', n);
            continue;
        end

        test_count = test_count + 1;
        fprintf('\n--- Test %d: %s (size %d) ---\n', test_count, filename, n);

        % Set up the problem: A * x = b where x_true = ones(n,1)
        x_true = ones(n, 1);
        b = A * x_true;

        fprintf('Problem setup: A * x = b, where x_true = ones(%d,1)\n', n);
        fprintf('||b|| = %.6f\n', norm(b));

        % Initialize result structure
        results(test_count).filename = filename;
        results(test_count).matrix_size = n;
        results(test_count).condition_number = cond(A);

        fprintf('\n=== Standard Cholesky Decomposition ===\n');

        % Test 1: Standard Cholesky decomposition
        try
            tic;
            L_chol = choleskyDecomposition(A);
            time_chol = toc;

            % Solve using Cholesky method
            tic;
            x_chol = choleskyMethod(A, b);
            solve_time_chol = toc;

            % Compute errors
            error_chol = norm(x_chol - x_true);
            relative_error_chol = error_chol / norm(x_true);
            residual_chol = norm(A * x_chol - b);
            relative_residual_chol = residual_chol / norm(b);

            % Verify decomposition
            A_reconstructed = L_chol * L_chol';
            decomp_error_chol = norm(A_reconstructed - A, 'fro') / norm(A, 'fro');

            fprintf('SUCCESS: Standard Cholesky completed\n');
            fprintf('  Decomposition time: %.4f seconds\n', time_chol);
            fprintf('  Solution time: %.4f seconds\n', solve_time_chol);
            fprintf('  Solution error: ||x_computed - x_true|| = %.2e\n', error_chol);
            fprintf('  Relative solution error: %.2e\n', relative_error_chol);
            fprintf('  Residual: ||A*x - b|| = %.2e\n', residual_chol);
            fprintf('  Relative residual: %.2e\n', relative_residual_chol);
            fprintf('  Decomposition error: %.2e\n', decomp_error_chol);

            % Store results
            results(test_count).chol_success = true;
            results(test_count).chol_decomp_time = time_chol;
            results(test_count).chol_solve_time = solve_time_chol;
            results(test_count).chol_solution_error = error_chol;
            results(test_count).chol_relative_error = relative_error_chol;
            results(test_count).chol_residual = residual_chol;
            results(test_count).chol_relative_residual = relative_residual_chol;
            results(test_count).chol_decomp_error = decomp_error_chol;

        catch ME
            fprintf('FAILED: Standard Cholesky decomposition\n');
            fprintf('  Error: %s\n', ME.message);
            results(test_count).chol_success = false;
            results(test_count).chol_error_msg = ME.message;
        end

        fprintf('\n=== Modified Cholesky Decomposition ===\n');

        % Test 2: Modified Cholesky decomposition
        try
            tic;
            [L_tilde, D] = modifiedCholeskyDecomposition(A);
            time_mod = toc;

            % Solve using modified Cholesky method
            tic;
            x_mod = modifiedCholeskyMethod(A, b);
            solve_time_mod = toc;

            % Compute errors
            error_mod = norm(x_mod - x_true);
            relative_error_mod = error_mod / norm(x_true);
            residual_mod = norm(A * x_mod - b);
            relative_residual_mod = residual_mod / norm(b);

            % Verify decomposition
            A_reconstructed_mod = L_tilde * D * L_tilde';
            decomp_error_mod = norm(A_reconstructed_mod - A, 'fro') / norm(A, 'fro');

            fprintf('SUCCESS: Modified Cholesky completed\n');
            fprintf('  Decomposition time: %.4f seconds\n', time_mod);
            fprintf('  Solution time: %.4f seconds\n', solve_time_mod);
            fprintf('  Solution error: ||x_computed - x_true|| = %.2e\n', error_mod);
            fprintf('  Relative solution error: %.2e\n', relative_error_mod);
            fprintf('  Residual: ||A*x - b|| = %.2e\n', residual_mod);
            fprintf('  Relative residual: %.2e\n', relative_residual_mod);
            fprintf('  Decomposition error: %.2e\n', decomp_error_mod);
            fprintf('  Smallest diagonal in D: %.2e\n', min(diag(D)));

            % Store results
            results(test_count).mod_success = true;
            results(test_count).mod_decomp_time = time_mod;
            results(test_count).mod_solve_time = solve_time_mod;
            results(test_count).mod_solution_error = error_mod;
            results(test_count).mod_relative_error = relative_error_mod;
            results(test_count).mod_residual = residual_mod;
            results(test_count).mod_relative_residual = relative_residual_mod;
            results(test_count).mod_decomp_error = decomp_error_mod;
            results(test_count).mod_min_diagonal = min(diag(D));

        catch ME
            fprintf('FAILED: Modified Cholesky decomposition\n');
            fprintf('  Error: %s\n', ME.message);
            results(test_count).mod_success = false;
            results(test_count).mod_error_msg = ME.message;
        end

        fprintf('\n=== MATLAB Built-in Comparison ===\n');

        % Compare with MATLAB's built-in methods
        try
            % Using backslash operator
            tic;
            x_matlab = A \ b;
            time_matlab = toc;

            error_matlab = norm(x_matlab - x_true);
            relative_error_matlab = error_matlab / norm(x_true);
            residual_matlab = norm(A * x_matlab - b);
            relative_residual_matlab = residual_matlab / norm(b);

            fprintf('MATLAB backslash operator:\n');
            fprintf('  Solution time: %.4f seconds\n', time_matlab);
            fprintf('  Solution error: %.2e\n', error_matlab);
            fprintf('  Relative solution error: %.2e\n', relative_error_matlab);
            fprintf('  Residual: %.2e\n', residual_matlab);
            fprintf('  Relative residual: %.2e\n', relative_residual_matlab);

            % Store MATLAB results
            results(test_count).matlab_success = true;
            results(test_count).matlab_time = time_matlab;
            results(test_count).matlab_solution_error = error_matlab;
            results(test_count).matlab_relative_error = relative_error_matlab;
            results(test_count).matlab_residual = residual_matlab;
            results(test_count).matlab_relative_residual = relative_residual_matlab;

        catch ME
            fprintf('FAILED: MATLAB backslash operator\n');
            fprintf('  Error: %s\n', ME.message);
            results(test_count).matlab_success = false;
            results(test_count).matlab_error_msg = ME.message;
        end

        fprintf('\n');

    catch ME
        fprintf('ERROR processing file %s: %s\n\n', filename, ME.message);
        continue;
    end

end

fprintf('========================================\n');
fprintf('=== SUMMARY OF ALL TESTS ===\n');
fprintf('========================================\n');

if test_count == 0
    fprintf('No matrices were successfully tested.\n');
    return;
end

fprintf('\nTest Summary:\n');
fprintf('%-15s %8s %12s %12s %12s\n', 'Matrix', 'Size', 'Cond(A)', 'Std Chol', 'Mod Chol');
fprintf('%-15s %8s %12s %12s %12s\n', repmat('-', 1, 15), repmat('-', 1, 8), repmat('-', 1, 12), repmat('-', 1, 12), repmat('-', 1, 12));

for i = 1:test_count
    r = results(i);
    std_status = 'FAIL';
    mod_status = 'FAIL';

    if isfield(r, 'chol_success') && r.chol_success
        std_status = 'SUCCESS';
    end

    if isfield(r, 'mod_success') && r.mod_success
        mod_status = 'SUCCESS';
    end

    fprintf('%-15s %8d %12.2e %12s %12s\n', ...
        r.filename(1:min(15, end)), r.matrix_size, r.condition_number, std_status, mod_status);
end

fprintf('\nDetailed Error Analysis (for successful tests):\n');
fprintf('%-15s %12s %12s %12s %12s\n', 'Matrix', 'Std relErr', 'Mod relErr', 'Std relRes', 'Mod relRes');
fprintf('%-15s %12s %12s %12s %12s\n', repmat('-', 1, 15), repmat('-', 1, 12), repmat('-', 1, 12), repmat('-', 1, 12), repmat('-', 1, 12));

for i = 1:test_count
    r = results(i);

    if isfield(r, 'chol_success') && r.chol_success && ...
            isfield(r, 'mod_success') && r.mod_success
        fprintf('%-15s %12.2e %12.2e %12.2e %12.2e\n', ...
            r.filename(1:min(15, end)), ...
            r.chol_relative_error, r.mod_relative_error, ...
            r.chol_relative_residual, r.mod_relative_residual);
    end

end
