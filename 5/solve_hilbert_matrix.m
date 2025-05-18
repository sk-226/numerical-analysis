function solve_hilbert_matrix(n)
    % SOLVE_HILBERT_MATRIX  Solve a Hilbert linear system and study its conditioning.
    %   solve_hilbert_matrix(N) builds the N-by-N Hilbert matrix A,
    %   computes the infinity‑norm condition number kappa_inf(A) = ‖A‖_∞ ‖A^{-1}‖_∞
    %   using a custom LU factorisation with partial pivoting,
    %   solves A x = b with b = A*ones(N,1),
    %   and prints the residual and error infinity norms.
    %
    %   This code is self‑contained (no calls to MATLAB's lu, inv, or backslash)
    %   so you can inspect numerical errors arising from the algorithm itself.
    %
    %   Example:
    %       >> solve_hilbert_matrix(10)

    % Validate input
    if nargin ~= 1 || ~isscalar(n) || n <= 0 || n ~= floor(n)
        error('Input must be a positive integer size.');
    end

    %----------------------------------------------------------------------
    % Build Hilbert matrix A and right‑hand side b
    %----------------------------------------------------------------------
    A = zeros(n);

    for i = 1:n

        for j = 1:n
            A(i, j) = 1 / (i + j - 1);
        end

    end

    true_x = ones(n, 1);
    b = A * true_x;

    %----------------------------------------------------------------------
    % Infinity‑norm of A
    %----------------------------------------------------------------------
    normInfA = max(sum(abs(A), 2));

    %----------------------------------------------------------------------
    % LU factorisation with partial pivoting
    %----------------------------------------------------------------------
    [L, U, P] = lu_pp(A);

    %----------------------------------------------------------------------
    % Compute x by forward/back substitution
    %----------------------------------------------------------------------
    y = forward_sub(L, P * b);
    x = back_sub(U, y);

    %----------------------------------------------------------------------
    % Build (or estimate) A^{-1} to get ‖A^{-1}‖_∞
    %   For moderate n we form it explicitly; for large n one could
    %   replace this with an iterative condition‑number estimator.
    %----------------------------------------------------------------------
    invA = zeros(n);
    I = eye(n);

    for j = 1:n
        % Solve A z = e_j
        yj = forward_sub(L, P * I(:, j));
        invA(:, j) = back_sub(U, yj);
    end

    normInfInvA = max(sum(abs(invA), 2));

    kappaInf = normInfA * normInfInvA;

    %----------------------------------------------------------------------
    % Residual and error
    %----------------------------------------------------------------------
    residual = b - A * x;
    resNormInf = norm(residual, Inf);
    errNormInf = norm(true_x - x, Inf);

    %----------------------------------------------------------------------
    % Print results
    %----------------------------------------------------------------------
    fprintf('n               = %d\n', n);
    fprintf('‖A‖_∞          = %.4e\n', normInfA);
    fprintf('‖A^{-1}‖_∞     = %.4e\n', normInfInvA);
    fprintf('κ_∞(A)          = %.4e\n', kappaInf);
    fprintf('Residual ‖b-Ax‖_∞ = %.4e\n', resNormInf);
    fprintf('Error ‖x-true‖_∞  = %.4e\n', errNormInf);

    %----------------------------------------------------------------------
    % Nested helper functions
    %----------------------------------------------------------------------

    function [L, U, P] = lu_pp(M)
        % LU factorisation with partial pivoting.
        k = size(M, 1);
        P = eye(k);
        L = eye(k);
        U = M;

        for i = 1:k - 1
            % pivot
            [~, pivot] = max(abs(U(i:k, i)));
            pivot = pivot + i - 1;

            if pivot ~= i
                U([i pivot], :) = U([pivot i], :);
                P([i pivot], :) = P([pivot i], :);

                if i > 1
                    L([i pivot], 1:i - 1) = L([pivot i], 1:i - 1);
                end

            end

            % elimination
            for j = i + 1:k
                L(j, i) = U(j, i) / U(i, i);
                U(j, i:k) = U(j, i:k) - L(j, i) * U(i, i:k);
            end

        end

    end

    function y = forward_sub(LGt, rhs)
        % Solve L y = rhs for lower‑triangular L with unit diagonal
        m = length(rhs);
        y = zeros(m, 1);

        for r = 1:m
            y(r) = rhs(r) - LGt(r, 1:r - 1) * y(1:r - 1);
        end

    end

    function x = back_sub(UT, rhs)
        % Solve U x = rhs for upper‑triangular U
        m = length(rhs);
        x = zeros(m, 1);

        for r = m:-1:1
            x(r) = (rhs(r) - UT(r, r + 1:end) * x(r + 1:end)) / UT(r, r);
        end

    end

end
