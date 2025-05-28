function [x] = choleskyMethod(A, b)
    % CHOLESKYMETHOD  Solve a linear system using Cholesky decomposition.
    %
    %   x = CHOLESKYMETHOD(A, b) solves the linear system A*x = b using Cholesky
    %   decomposition, where A is a symmetric positive-definite matrix. The
    %   solution is obtained by forward and backward substitution.
    %
    %   INPUT
    %     A : n×n symmetric positive-definite matrix (coefficient matrix)
    %     b : n×1 vector (right-hand side vector)
    %
    %   OUTPUT
    %     x : n×1 solution vector
    %
    %   NOTES
    %   • The coefficient matrix A must be symmetric and positive-definite.
    %   • The algorithm first performs Cholesky decomposition A = L*L^T,
    %     then solves L*y = b by forward substitution, and finally
    %     solves L^T*x = y by backward substitution.
    %
    %   EXAMPLES
    %     >> A = [4, 2; 2, 5];  b = [6; 7];
    %     >> x = choleskyMethod(A, b);
    %     >> norm(A*x - b)                    % should be near machine epsilon
    % -------------------------------------------------------------------------

    % --------------------------- input validation ------------------------
    if ~ismatrix(A) || size(A, 1) ~= size(A, 2)
        error('CHOLESKYMETHOD:NotSquare', ...
        'Coefficient matrix A must be square.');
    end

    n = size(A, 1);

    if size(b, 1) ~= n || size(b, 2) ~= 1
        error('CHOLESKYMETHOD:IncompatibleDimensions', ...
        'Right-hand side vector b has incorrect dimensions.');
    end

    % ------------------------ Cholesky decomposition ---------------------
    L = choleskyDecomposition(A);

    % --------------------- forward and backward substitution -------------
    % solve L * y = b by forward substitution
    y = forwardSubstitution(L, b);

    % solve L^T * x = y by backward substitution
    x = backwardSubstitution(L', y);
end

% -------------------------------------------------------------------------
% Helper functions
% -------------------------------------------------------------------------

function [y] = forwardSubstitution(L, b)
    % FORWARDSUBSTITUTION  Solve L*y = b by forward substitution.
    %
    %   y = FORWARDSUBSTITUTION(L, b) solves the lower triangular system L*y = b
    %   using forward substitution.
    %
    %   INPUT
    %     L : n×n lower triangular matrix
    %     b : n×1 right-hand side vector
    %
    %   OUTPUT
    %     y : n×1 solution vector

    n = size(L, 1);
    y = zeros(n, 1);

    for i = 1:n
        sum_term = 0;

        for j = 1:i - 1
            sum_term = sum_term + L(i, j) * y(j);
        end

        y(i) = (b(i) - sum_term) / L(i, i);
    end

end

function [x] = backwardSubstitution(U, y)
    % BACKWARDSUBSTITUTION  Solve U*x = y by backward substitution.
    %
    %   x = BACKWARDSUBSTITUTION(U, y) solves the upper triangular system U*x = y
    %   using backward substitution.
    %
    %   INPUT
    %     U : n×n upper triangular matrix
    %     y : n×1 right-hand side vector
    %
    %   OUTPUT
    %     x : n×1 solution vector

    n = size(U, 1);
    x = zeros(n, 1);

    for i = n:-1:1
        sum_term = 0;

        for j = i + 1:n
            sum_term = sum_term + U(i, j) * x(j);
        end

        x(i) = (y(i) - sum_term) / U(i, i);
    end

end
