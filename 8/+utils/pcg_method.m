function [x, results] = pcg_method(A, preconditioner, opt)

    arguments
        A (:, :) {mustBeSquare} % must be SPD
        preconditioner (1, 1) {mustBeFunctionHandle} % preconditioner
        opt.tol (1, 1) {mustBePositive} = 1.0e-10
        opt.max_iter (1, 1) {mustBePositive, mustBeInteger} = []
    end

    % default value of max_iter
    if isempty(opt.max_iter)
        opt.max_iter = 2 * size(A, 1);
    end

    % options
    tol = opt.tol; % tolerance (epsilon)
    max_iter = opt.max_iter; % maximum number of iterations

    % ------------------- INITIALIZE -------------------

    % size of A
    n = size(A, 1);

    % true solution
    x_true = ones(n, 1);

    % right-hand side vector
    b = A * x_true;

    % initial guess: all zeros
    x = zeros(n, 1);

    % store relative residual 2-norm, relative error 2-norm, relative error A-norm
    % check: these histories are vectors of (max_iter + 1) elements
    hist_relres_2 = zeros(max_iter + 1, 1); % ||r^{(k)}||_2 / ||b||_2
    hist_relerr_2 = zeros(max_iter + 1, 1); % ||x_true - x^{(k)}||_2 / ||x_true||_2
    hist_relerr_A = zeros(max_iter + 1, 1); % ||x_true - x^{(k)}||_A / ||x_true||_A

    % start timer
    % caution: the time includes the time for calculating and storing the histories
    tic;

    % 2-norm of b, x_true and A-norm of x_true
    norm2_b = norm(b);
    norm2_x_true = norm(x_true);
    normA_x_true = sqrt(x_true' * A * x_true);

    % set initial residual
    r = b - A * x;

    % calculate error between true solution and "initial" guess
    % used for calculating relative errors 2-norm and A-norm
    err = x_true - x; % e^{(0)} = x_true - x^{(0)}

    % store "1st" relative residual 2-norm, relative error 2-norm, relative error A-norm
    hist_relres_2(1) = norm(r) / norm2_b;
    hist_relerr_2(1) = norm(err) / norm2_x_true;
    hist_relerr_A(1) = sqrt(err' * A * err) / normA_x_true;

    % preconditioning
    z = preconditioner(r);

    % set initial search direction
    p = z; % p^{(0)} = z^{(0)}

    % calculate 1st rho_old
    % this is for the efficient calculation
    rho_old = r' * z;

    % ------------------- ITERATION -------------------

    % start iteration
    for iter = 1:max_iter

        % calculate A * p
        % this is for the efficient calculation
        w = A * p;

        % calculate sigma
        sigma = p' * w;

        % calculate alpha
        alpha = rho_old / sigma;

        % update approximate solution
        x = x + alpha * p;

        % update residual
        r = r - alpha * w;

        % calculate error between true solution and approximate solution
        err = x_true - x;

        % store relative residual 2-norm, relative error 2-norm, relative error A-norm
        hist_relres_2(iter + 1) = norm(r) / norm2_b;
        hist_relerr_2(iter + 1) = norm(err) / norm2_x_true;
        hist_relerr_A(iter + 1) = sqrt(err' * A * err) / normA_x_true;

        % check convergence
        if hist_relres_2(iter + 1) < tol
            is_converged = true;
            fprintf('... Converged! (iter = %d)\n', iter);
            fprintf('\n');
            break;
        end

        % Update z (preconditioning)
        z = preconditioner(r);

        % calculate rho_new for beta's numerator
        % after beta's calculation, update rho_old for next iteration with this rho_new
        rho_new = r' * z; % (r^{(k+1)}, z^{(k+1)})

        % calculate beta
        beta = rho_new / rho_old; % = (r_new, z_new) / (r_old, z_old)

        % update rho_old for next iteration
        rho_old = rho_new; % = (r_new, z_new) where r_new: r^{(k+1)} (z_new: z^{(k+1)})

        % update search direction
        % check: Orthogonal Residual condition
        p = z + beta * p; % p^{(k+1)} = z^{(k+1)} + beta * p^{(k)}

    end % end of iteration

    % ------------------- END -------------------

    % end timer
    time = toc;

    % check if not converged
    % if not converged, print message (I think it's not necessary; just for printing the message)
    if hist_relres_2(iter + 1) >= tol
        is_converged = false;
        fprintf('... NOT converged. (iter = %d)\n', iter);
        fprintf('\n');
    end

    % true relative residual 2-norm
    % check: residual gap
    true_relres2 = norm(b - (A * x)) / norm2_b;

    % store results (return value)
    results = struct( ...
        'iter_final', {iter}, ...
        'is_converged', {is_converged}, ... % boolean
        'time', {time}, ...
        'hist_relres_2', {hist_relres_2(1:iter + 1)}, ...
        'true_relres_2', {true_relres2}, ...
        'hist_relerr_2', {hist_relerr_2(1:iter + 1)}, ...
        'hist_relerr_A', {hist_relerr_A(1:iter + 1)} ...
    );
end
