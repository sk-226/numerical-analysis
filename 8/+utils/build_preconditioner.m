function preconditioner = build_preconditioner(A, type, opts)
    % BUILDPRECONDITIONER  Construct a function handle that applies a
    %                      preconditioner (computes M^{-1} r)
    %
    %   This function is used in pcg_method.m
    %
    %   INPUT
    %       A     : SPD matrix (sparse recommended)
    %       type  : 'diag' | 'ssor' | 'ic' | 'diag+ssor' | 'diag+ic'
    %       opts  : Parameters such as ω, droptol (optional)
    %
    %   OUTPUT
    %       preconditioner : function_handle  –  r := M^{-1} r
    %
    %   NOTE: If you want to measure the setup time, do so at the call site:
    %       tic;  prec = build_preconditioner(...);  setupTime = toc;
    %
    %   EXAMPLES
    %     >> A = Problem.A; % load problem
    %     >> preconditioner = build_preconditioner(A, 'ic'); % Incomplete Cholesky
    %     >> [x, results] = pcg_method(A, preconditioner, tol = 1.0e-12, max_iter = 2 * n);
    % -------------------------------------------------------------------------

    % --------------------------- input validation ----------------------------
    arguments
        A (:, :) % mustBeSymmetric
        type string
        opts.omega (1, 1) {mustBeInRange(opts.omega, 0, 2, "exclusive")} = 1.0 % SSOR: over-relaxation parameter
        opts.ictype {mustBeMember(opts.ictype, ['nofill', 'ict'])} = 'nofill' % ichol: type (default: "nofill" / "ict")
        opts.droptol (1, 1) {mustBeNonnegative(opts.droptol)} = 0.0 % ichol: drop tolerance
    end

    % check if A is symmetric
    % TODO: mustBeSymmetric ないのえぐいって
    if ~issymmetric(A)
        error("A must be symmetric");
    end
    % --------------------------- end input validation ------------------------

    % size of A
    n = size(A, 1);

    switch lower(type) % convert to lowercase (case insensitive!!)
        case "diag" % diagonal scaling
            d_inv = 1 ./ diag(A); % diagonal vector
            preconditioner = @(r) d_inv .* r; % D^{-1} * r

        case "ssor" % SSOR
            omega = opts.omega;
            d = diag(A); % diagonal vector
            L = sparse(tril(A, -1)); % strictly lower triangular matrix (U = L' because A is symmetric)
            DL = spdiags(d, 0, n, n) + (omega * L); % (D + omega * L): lower triangular matrix
            DU = spdiags(d, 0, n, n) + (omega * L'); % (D + omega * U): upper triangular matrix
            % NOTE: same as the following (might be faster and compact?? like backslash)
            %       y = mldivide(DL,r); (y := DL^{-1} * r)
            %       w = d .* y; (w := D * y)
            %       z = mldivide(DU,w); (z := DU^{-1} * w)
            %       z := M^{-1} * r = DU^{-1} * ( D * ( DL^{-1} * r ) )
            preconditioner = @(r) mldivide(DU, d .* mldivide(DL, r));

        case "ic" % Incomplete Cholesky
            L = ichol(A, struct("type", opts.ictype, "droptol", opts.droptol));
            % NOTE: same as the following (might be faster and compact?? like backslash)
            %       y = mldivide(L,r);
            %       z = mldivide(L',y);
            preconditioner = @(r) mldivide(L', mldivide(L, r));

        case "diag+ssor" % diagonal scaling -> SSOR
            S = spdiags(1 ./ sqrt(diag(A)), 0, n, n);
            Atil = S * A * S;
            omega = opts.omega;
            Dtil = spdiags(diag(Atil), 0, n, n);
            Ltil = sparse(tril(Atil, -1));
            T1 = sparse(Dtil / omega + Ltil);
            T2 = sparse(Dtil * (2 - omega) / omega);
            SSOR = @(v) (T1') \ (T2 \ (T1 \ v));
            preconditioner = @(r) S * SSOR(S * r);

        case "diag+ic" % diagonal scaling -> IC
            % NOTE: It is effective when using ict and setting droptol for ichol.
            %       No huge changes in performance if using nofill. (probably??)
            %       Nofill is the memory efficient option,
            %       but slower convergence compared to ict. (probably??)ke
            
            S = spdiags(1 ./ sqrt(diag(A)), 0, n, n);
            Atil = S * A * S;
            L = ichol(Atil, struct("type", opts.ictype, "droptol", opts.droptol));
            ICCG = @(v) L' \ (L \ v);
            preconditioner = @(r) S * ICCG(S * r);

        otherwise
            error("Unknown preconditioner type: %s", type);
    end

end
