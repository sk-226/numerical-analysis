function solve_matrix(A)
    % SOLVE_MATRIX  Solve a linear system with a (symmetric) positive‑definite matrix
    %              and study its infinity‑norm conditioning.
    %
    %   solve_matrix(A) takes a *square numeric* matrix A (typically real SPD),
    %   forms the right‑hand side b = A*ones(n,1), solves A x = b with a *custom*
    %   LU factorisation (partial pivoting), and reports
    %       ‖A‖_∞ , ‖A⁻¹‖_∞ , κ_∞(A) , residual , error .
    %
    %   Example
    %   -------
    %       >> n = 8;  A = diag(2*ones(n,1)) + diag(-1*ones(n-1,1),1) + diag(-1*ones(n-1,1),-1);
    %       >> solve_matrix(A)                    % SPD tridiagonal (Poisson 1‑D)
    %
    %   See also IS_SPD          (for SPD checking, optional)
    % -------------------------------------------------------------------------

    % --------------------------- input checks ----------------------------
    if nargin ~= 1 || ~isnumeric(A) || ~ismatrix(A)
        error('Input must be a numeric matrix.');
    end

    [n, m] = size(A);

    if n ~= m
        error('Matrix must be square; received %d-by-%d.', n, m);
    end

    % ---------------- SPD check ----------------
    if ~is_spd(A, 'chol')
        warning('SOLVE_MATRIX:NotSPD', 'A does not appear to be SPD; results may be meaningless.');
    end

    % --------------------------- build RHS -------------------------------
    true_x = ones(n, 1);
    b = A * true_x;

    % ------------------ infinity‑norm of A (row sums) --------------------
    normInfA = max(sum(abs(A), 2));

    % ------------------ LU factorisation with pivoting -------------------
    [L, U, P] = lu_pp(A);

    % ------------------ forward & back substitutions ---------------------
    y = forward_sub(L, P * b);
    x = back_sub(U, y);

    % ------------------ explicit inverse for ‖A⁻¹‖_∞ ---------------------
    invA = zeros(n);
    I = eye(n);

    for j = 1:n
        yj = forward_sub(L, P * I(:, j));
        invA(:, j) = back_sub(U, yj);
    end

    normInfInvA = max(sum(abs(invA), 2));

    kappaInf = normInfA * normInfInvA;

    % ------------------ residual & forward error -------------------------
    residual = b - A * x;
    resNormInf = norm(residual, Inf);
    errNormInf = norm(true_x - x, Inf);

    % ---------------------------- print ----------------------------------
    fprintf('n                    = %d\n', n);
    fprintf('‖A‖_∞                = %.4e\n', normInfA);
    fprintf('‖A^{-1}‖_∞           = %.4e\n', normInfInvA);
    fprintf('κ_∞(A)               = %.4e\n', kappaInf);
    fprintf('Residual ‖b-Ax‖_∞    = %.4e\n', resNormInf);
    fprintf('Error ‖x-true‖_∞     = %.4e\n', errNormInf);

    %----------------------------------------------------------------------
    % Helper functions
    %----------------------------------------------------------------------
    function [L, U, P] = lu_pp(M)
        % LU factorisation with partial pivoting.
        n = size(M, 1);
        P = eye(n);
        L = eye(n);
        U = M;


        for i = 1:n - 1
            % pivot row
            [~, pivot] = max(abs(U(i:n, i)));
            pivot = pivot + i - 1;


            if pivot ~= i
                U([i pivot], :) = U([pivot i], :);
                P([i pivot], :) = P([pivot i], :);
                if i > 1, L([i pivot], 1:i - 1) = L([pivot i], 1:i - 1); end
            end

            % elimination
            for j = i + 1:n
                L(j, i) = U(j, i) / U(i, i);
                U(j, i:n) = U(j, i:n) - L(j, i) * U(i, i:n);
            end

        end

    end

    function y = forward_sub(Lt, rhs)
        % Solve L y = rhs  (unit‑lower‑triangular L).
        m = length(rhs);
        y = zeros(m, 1);

        for r = 1:m
            y(r) = rhs(r) - Lt(r, 1:r - 1) * y(1:r - 1);
        end

    end

    function x = back_sub(Ut, rhs)
        % Solve U x = rhs  (upper‑triangular U).
        m = length(rhs);
        x = zeros(m, 1);

        for r = m:-1:1
            x(r) = (rhs(r) - Ut(r, r + 1:end) * x(r + 1:end)) / Ut(r, r);
        end

    end

end
