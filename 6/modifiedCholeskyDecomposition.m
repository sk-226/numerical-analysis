function [L_tilde, D] = modifiedCholeskyDecomposition(A)
    % MODIFIEDCHOLESKYDECOMPOSITION  Perform modified Cholesky decomposition without square roots.
    %
    %   [L_tilde, D] = MODIFIEDCHOLESKYDECOMPOSITION(A) decomposes the symmetric
    %   positive-definite matrix A into the form A = L_tilde * D * L_tilde^T,
    %   where L_tilde is a unit lower triangular matrix and D is a diagonal matrix.
    %
    %   INPUT
    %     A : n×n symmetric positive-definite matrix
    %
    %   OUTPUT
    %     L_tilde : n×n unit lower triangular matrix (diagonal elements = 1)
    %     D       : n×n diagonal matrix
    %
    %   NOTES
    %   • This is an improved version of standard Cholesky decomposition that
    %     avoids square root computations, which are more expensive than basic
    %     arithmetic operations.
    %
    %   EXAMPLES
    %     >> A = [4, 2; 2, 5];
    %     >> [L_tilde, D] = modifiedCholeskyDecomposition(A);
    %     >> norm(A - L_tilde * D * L_tilde', 'fro')   % should be near machine epsilon
    % -------------------------------------------------------------------------

    % --------------------------- input validation ------------------------
    if ~ismatrix(A) || size(A, 1) ~= size(A, 2)
        error('MODIFIEDCHOLESKYDECOMPOSITION:NotSquare', ...
        'Input matrix A must be square.');
    end

    % symmetry check
    if ~isequal(A, A')
        error('MODIFIEDCHOLESKYDECOMPOSITION:NotSymmetric', ...
        'Input matrix A must be symmetric.');
    end

    n = size(A, 1);
    L_tilde = eye(n); % identity matrix initialization (diagonal elements = 1)
    D = zeros(n, n); % diagonal matrix

    % ------------- modified Cholesky decomposition algorithm -------------
    % set d_11 = a_11, l_tilde_11 = 1 as per algorithm
    D(1, 1) = A(1, 1);

    for i = 2:n
        % L_tilde(i, i) = 1 is already set

        for j = 1:i
            % s = sum_{k=1}^{j-1} L_tilde(i, k) * D(k, k) * L_tilde(j, k)
            s = 0;

            for k = 1:j - 1
                s = s + L_tilde(i, k) * D(k, k) * L_tilde(j, k);
            end

            if i == j
                % diagonal case: d_ii = a_ii - s
                D(i, i) = A(i, i) - s;

                % positive-definiteness check
                if D(i, i) <= 0
                    error('MODIFIEDCHOLESKYDECOMPOSITION:NotPositiveDefinite', ...
                    'Matrix is not positive-definite. Modified Cholesky decomposition failed.');
                end

            else
                % off-diagonal case: L_tilde(i, j) = (a_ij - s) / d_jj
                L_tilde(i, j) = (A(i, j) - s) / D(j, j);
            end

        end

    end

end
