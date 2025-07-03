function [L, D, deltaA] = incompleteCholeskyDecomposition(A)
    % INCOMPLETECHOLESKYDECOMPOSITION  Perform incomplete Cholesky decomposition.
    %
    %   [L, D, deltaA] = INCOMPLETECHOLESKYDECOMPOSITION(A) decomposes the
    %   symmetric matrix A into the form A = L * D * L^T + deltaA,
    %   where L is a unit lower triangular matrix with the same sparsity
    %   pattern as A, D is a diagonal matrix, and deltaA is the error matrix.
    %
    %   INPUT
    %     A : n×n symmetric matrix (typically sparse)
    %
    %   OUTPUT
    %     L      : n×n unit lower triangular matrix (diagonal elements = 1)
    %              with the same sparsity pattern as the lower triangular part of A
    %     D      : n×n diagonal matrix
    %     deltaA : n×n error matrix such that A = L * D * L^T + deltaA
    %
    %   NOTES
    %   - This method preserves the sparsity pattern of the original matrix A
    %   - Only components with indices in P = {(i,j) | a_ij ≠ 0} are computed
    %   - Other components are forced to be zero, resulting in an incomplete decomposition
    %   - The error deltaA arises from this sparsity constraint
    %   - As the ichol function, no need to return deltaA
    %
    %   EXAMPLES
    %     >> A = [4, 2, 0; 2, 5, 1; 0, 1, 3];  % sparse symmetric matrix
    %     >> [L, D, deltaA] = incompleteCholeskyDecomposition(A);
    %     >> norm(A - (L * D * L' + deltaA), 'fro')   % should be near machine epsilon
    % -------------------------------------------------------------------------

    % --------------------------- input validation ------------------------
    if ~ismatrix(A) || size(A, 1) ~= size(A, 2)
        error('INCOMPLETECHOLESKYDECOMPOSITION:NotSquare', ...
        'Input matrix A must be square.');
    end

    % symmetry check
    if ~isequal(A, A')
        error('INCOMPLETECHOLESKYDECOMPOSITION:NotSymmetric', ...
        'Input matrix A must be symmetric.');
    end

    n = size(A, 1);
    L = eye(n); % identity matrix initialization (diagonal elements = 1)
    D = zeros(n, n); % diagonal matrix

    % determine sparsity pattern P = {(i,j) | a_ij ≠ 0}
    % for lower triangular part only
    P = (tril(A) ~= 0);

    % ------------- incomplete Cholesky decomposition algorithm -------------
    % First, perform complete modified Cholesky decomposition
    
    % set d_11 = a_11
    D(1, 1) = A(1, 1);
    
    % check for positive-definiteness of first element
    if D(1, 1) <= 0
        error('INCOMPLETECHOLESKYDECOMPOSITION:NotPositiveDefinite', ...
        'Matrix is not positive-definite. First diagonal element is non-positive.');
    end

    for i = 2:n
        % L(i, i) = 1 is already set

        for j = 1:i
            % s = sum_{k=1}^{j-1} L(i, k) * D(k, k) * L(j, k)
            % compute full sum without sparsity constraint
            s = 0;

            for k = 1:j - 1
                s = s + L(i, k) * D(k, k) * L(j, k);
            end

            if i == j
                % diagonal case: d_ii = a_ii - s
                D(i, i) = A(i, i) - s;

                % positive-definiteness check
                if D(i, i) <= 0
                    error('INCOMPLETECHOLESKYDECOMPOSITION:NotPositiveDefinite', ...
                    'Matrix is not positive-definite. Diagonal element %d is non-positive.', i);
                end

            else
                % off-diagonal case: L(i, j) = (a_ij - s) / d_jj
                if P(i, j) == 1
                    % only compute if (i,j) is in the sparsity pattern P
                    L(i, j) = (A(i, j) - s) / D(j, j);
                else
                    % force to zero if not in sparsity pattern
                    L(i, j) = 0;
                end
            end

        end

    end

    % compute error matrix deltaA = A - L * D * L^T
    deltaA = A - L * D * L';

end


