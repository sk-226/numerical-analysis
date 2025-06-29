function plot_conv_hist(results, opt)
    arguments
        results
        opt.problem_name = []
        opt.y_lim = [1e-13 1e+1]
        opt.y_tick = [1e-12 1e-09 1e-06 1e-3 1]
        opt.y_tick_label = {'-12', '-9', '-6', '-3', '0'}

    end
    % plot convergence history
    % CAUTION: x_axis starts from 0
    x_axis = 0:results.iter_final;
    hold on, grid on;
    plot(x_axis, results.hist_relres_2, '-*', 'DisplayName', strcat('||r_k||_2/||b||_2'));
    plot(x_axis, results.hist_relerr_2, '-*', 'DisplayName', strcat('||e_k||_2/||x_{true}||_2'));
    plot(x_axis, results.hist_relerr_A, '-*', 'DisplayName', strcat('||e_k||_A/||x_{true}||_A'));
    legend, box on;
    if ~isempty(opt.problem_name)
        title(opt.problem_name);
    end
    xlabel('Number of Iterations');
    ylabel('Log_{10} of relative norm');
    ylim(gca, opt.y_lim);
    set(gca, ...
        'FontSize', 16, ...
        'YScale', 'log', ...
        'YTick', opt.y_tick, ...
        'YTickLabel', opt.y_tick_label);
    hold off;
end
