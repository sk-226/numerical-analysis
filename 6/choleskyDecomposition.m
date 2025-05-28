function [L] = choleskyDecomposition(A)
    % CHOLESKYDECOMPOSITION  Perform Cholesky decomposition of a symmetric positive-definite matrix.
    %
    %   L = CHOLESKYDECOMPOSITION(A) decomposes the symmetric positive-definite
    %   matrix A into the form A = L * L^T, where L is a lower triangular matrix.
    %
    %   INPUT
    %     A : n×n symmetric positive-definite matrix
    %
    %   OUTPUT
    %     L : n×n lower triangular matrix such that A = L * L^T
    %
    %   NOTES
    %   • The input matrix A must be symmetric and positive-definite.
    %   • If A is not positive-definite, the algorithm will fail when attempting
    %     to compute the square root of a non-positive diagonal element.
    %
    %   EXAMPLES
    %     >> A = [4, 2; 2, 5];
    %     >> L = choleskyDecomposition(A);
    %     >> norm(L * L' - A, 'fro')          % should be near machine epsilon
    % -------------------------------------------------------------------------

    % --------------------------- input validation ------------------------
    if ~ismatrix(A) || size(A, 1) ~= size(A, 2)
        error('CHOLESKYDECOMPOSITION:NotSquare', ...
        'Input matrix A must be square.');
    end

    % symmetry check
    if ~isequal(A, A')
        error('CHOLESKYDECOMPOSITION:NotSymmetric', ...
        'Input matrix A must be symmetric.');
    end

    n = size(A, 1);
    L = zeros(n, n);

    % -------------------- Cholesky decomposition algorithm ----------------
    for i = 1:n

        for j = 1:i

            if i == j
                % diagonal element computation
                % l_ii = sqrt(a_ii - sum_{k=1}^{i-1} l_ik^2)
                sum_squares = 0;

                for k = 1:i - 1
                    sum_squares = sum_squares + L(i, k) ^ 2;
                end

                diagonal_value = A(i, i) - sum_squares;

                % positive-definiteness check
                if diagonal_value <= 0
                    error('CHOLESKYDECOMPOSITION:NotPositiveDefinite', ...
                    'Matrix is not positive-definite. Cholesky decomposition failed.');
                end

                L(i, j) = sqrt(diagonal_value);
            else
                % off-diagonal element computation
                % l_ij = (1/l_jj) * (a_ij - sum_{k=1}^{j-1} l_ik * l_jk)
                sum_products = 0;

                for k = 1:j - 1
                    sum_products = sum_products + L(i, k) * L(j, k);
                end

                L(i, j) = (A(i, j) - sum_products) / L(j, j);
            end

        end

    end

end
