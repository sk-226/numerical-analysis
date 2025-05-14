function tf = is_spd(A, method, symTol, posTol)
    %IS_SPD  Test whether a real symmetric matrix is symmetric positive definite.
    %
    %   tf = IS_SPD(A)              –– uses Cholesky (default), tolerance eps (machine epsilon)
    %   tf = IS_SPD(A, METHOD)      –– METHOD = 'chol' (fast, default) | 'eig' (robust)
    %   tf = IS_SPD(A, METHOD, SYMTOL, POSTOL) –– user‑supplied tolerances
    %
    %   INPUT
    %     A      : real square matrix to test
    %     METHOD : 'chol' (fast, default)  |  'eig' (robust)
    %     SYMTOL : non‑negative scalar, treats ‖A-A.'‖_∞ ≤ symTol as symmetric
    %     POSTOL : non‑negative scalar, treats min(eig(A)) > posTol as SPD
    %
    %   OUTPUT
    %     tf     : logical scalar, true  ⇔  A is symmetric positive definite
    %
    %   NOTES
    %   • SPD ⇒ all eigenvalues are strictly > 0  ⇔  a Cholesky factor exists.
    %   • Cholesky is ≈ 1/2 the flops of full eigen‑decomposition and aborts
    %     early on a non‑positive pivot, but may throw an error for
    %     ill‑conditioned SPD matrices (round‑off).  The eigenvalue route is
    %     slower but gives a continuous measure (min eig(A)).
    %   • The symmetry test is separated from the SPD test so that numerical
    %     “nearly symmetric” inputs can be handled gracefully.
    %
    %   EXAMPLES
    %     >> is_spd(eye(5))                    % true  (chol, default)
    %     >> is_spd(hilb(10),'eig')            % true  (slow but robust)
    %     >> A = gallery('lehmer',8); A(1,8)=A(1,8)+1e‑8;
    %     >> is_spd(A,'chol')                  % false (fails symmetry test)
    %     >> is_spd(A,'chol',1e‑7)             % true  (looser symmetry tol)
    %
    %   AUTHOR  Suguru Kurita, Tokyo City University / May 2025
    % -------------------------------------------------------------------------
    

    % -------- defaults & simple input checks ----------------------------
    if nargin < 2 || isempty(method),  method  = 'chol';                   end
    if nargin < 3 || isempty(symTol),  symTol  = eps;                      end
    if nargin < 4 || isempty(posTol),  posTol  = eps;                      end
    
    if ~isreal(A)
        error('IS_SPD:InputNotReal','A must be real (Hermitian not supported).');
    end
    if size(A, 1) ~= size(A, 2)
        error('IS_SPD:NotSquare','A must be a square matrix.');
    end
    
    % --------------------------- symmetry test --------------------------
    nsym = norm(A - A.', inf);
    if nsym > symTol
        fprintf('is_spd: FAILED  -- not symmetric (‖A-A.''‖_inf = %.3g > %.3g)\n',...
                nsym, symTol);
        tf = false;
        return;
    end
    
    % ---------------------------- SPD test ------------------------------
    switch lower(method)
        case 'chol'                                % ---------- fast ----------
            [~, p] = chol(A);
            if p == 0
                tf = true;
            else
                fprintf('is_spd: FAILED  -- Cholesky breakdown at pivot %d\n',p);
                tf = false;
            end

        case 'eig'                                 % ---------- robust --------
            lam_min = min(eig(A,'vector'));
            if lam_min > posTol
                tf = true;
            else
                fprintf(['is_spd: FAILED  -- min eig = %.3g  ≤  posTol = %.3g ',...
                            '(likely ill-conditioned SPD)\n'], lam_min, posTol);
                tf = false;
            end

        otherwise
            error('IS_SPD:BadMethod','Unknown METHOD ''%s''.',method);
    end
end
