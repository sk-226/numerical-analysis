function print_num_results(results, opt)
    arguments
        results
        opt.problem_name = []
    end

    % CAUTION: final result index of hist_* is (iter_final+1)
    fprintf('========================== \n');
    fprintf('Numerical Results. \n');
    if ~isempty(opt.problem_name)
        fprintf('Problem: %s \n', opt.problem_name);
    end
    fprintf('========================== \n');

    if results.is_converged
        fprintf('Converged! (iter = %d)\n', results.iter_final);
    else
        fprintf('NOT converged. (max_iter = %d)\n', results.iter_final);
    end

    fprintf('# Iter.: %d\n', results.iter_final);
    fprintf('Time[s]: %.3f\n', results.time);
    fprintf('Relres_2norm = %.2e\n', results.hist_relres_2(results.iter_final + 1));
    fprintf('True_Relres_2norm = %.2e\n', results.true_relres_2);
    fprintf('Relerr_2norm = %.2e\n', results.hist_relerr_2(results.iter_final + 1));
    fprintf('Relerr_Anorm = %.2e\n', results.hist_relerr_A(results.iter_final + 1));
    fprintf('========================== \n');
    fprintf('\n');
end
