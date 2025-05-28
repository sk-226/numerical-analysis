function [x] = modifiedCholeskyMethod(A, b)
    % MODIFIEDCHOLESKYMETHOD  Solve a linear system using modified Cholesky decomposition.
    %
    %   x = MODIFIEDCHOLESKYMETHOD(A, b) solves the linear system A*x = b using
    %   modified Cholesky decomposition, where A is a symmetric positive-definite
    %   matrix. The solution is obtained by forward and backward substitution.
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
    %   • The algorithm performs modified Cholesky decomposition A = L̃*D*L̃^T,
    %     then solves the system through three steps:
    %     1) L̃*z = b by forward substitution
    %     2) D*y = z by diagonal matrix division
    %     3) L̃^T*x = y by backward substitution
    %
    %   EXAMPLES
    %     >> A = [4, 2; 2, 5];  b = [6; 7];
    %     >> x = modifiedCholeskyMethod(A, b);
    %     >> norm(b - A*x)                    % should be near machine epsilon
    % -------------------------------------------------------------------------

    % --------------------------- input validation ------------------------
    if ~ismatrix(A) || size(A, 1) ~= size(A, 2)
        error('MODIFIEDCHOLESKYMETHOD:NotSquare', ...
        'Coefficient matrix A must be square.');
    end

    n = size(A, 1);

    if size(b, 1) ~= n || size(b, 2) ~= 1
        error('MODIFIEDCHOLESKYMETHOD:IncompatibleDimensions', ...
        'Right-hand side vector b has incorrect dimensions.');
    end

    % -------------------- modified Cholesky decomposition ----------------
    [L_tilde, D] = modifiedCholeskyDecomposition(A);

    % -------------- solve L̃*D*L̃^T*x = b in three steps ----------------
    % step 1: solve L̃*z = b by forward substitution
    z = forwardSubstitution(L_tilde, b);

    % step 2: solve D*y = z (diagonal matrix system)
    y = diagonalSolve(D, z);

    % step 3: solve L̃^T*x = y by backward substitution
    x = backwardSubstitution(L_tilde', y);
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

function [y] = diagonalSolve(D, z)
    % DIAGONALSOLVE  Solve diagonal matrix equation D*y = z.
    %
    %   y = DIAGONALSOLVE(D, z) solves the diagonal system D*y = z, where D is
    %   a diagonal matrix.
    %
    %   INPUT
    %     D : n×n diagonal matrix
    %     z : n×1 right-hand side vector
    %
    %   OUTPUT
    %     y : n×1 solution vector

    n = size(D, 1);
    y = zeros(n, 1);

    for i = 1:n
        y(i) = z(i) / D(i, i);
    end

end
