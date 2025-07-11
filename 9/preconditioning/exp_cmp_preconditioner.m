clc; clearvars; close all;

import krylov.cg_method;
import krylov.build_preconditioner;
import krylov.pcg_method;
import utils.print_num_results;
import utils.plot_conv_hist;
import utils.prec_config2str;
import utils.expand_results_table;

% *IMPORTANT: MAKE SURE TO MKDIR BEFORE RUNNING THIS SCRIPT
exp_title = "exp_cmp_preconditioner";
inputs_dir = strcat("inputs/", exp_title);
outputs_dir = strcat("outputs/", exp_title);

if ~exist(inputs_dir, "dir")
    mkdir(inputs_dir);
end

if ~exist(outputs_dir, "dir")
    mkdir(outputs_dir);
end

d = dir(fullfile(inputs_dir,"*.mat"));
matrix_files = string({d.name}); 

% preconditioner configurations
preconditioner_configs = { ...
    struct('type', 'none'), ...
    struct('type', 'diag'), ...
    struct('type', 'ssor', 'omega', 1.0), ...
    struct('type', 'ic', 'ictype', 'nofill', 'droptol', 0.0), ...
};

% ======

% initialize results summary table
results_summary = table();

for i = 1:numel(matrix_files)
    % load problem
    load(fullfile(inputs_dir, matrix_files(i)));
    A = sparse(Problem.A);
    n = size(A, 1);

    tol = 1.0e-12;
    max_iter = 2 * n;

    prec_labels = strings(size(preconditioner_configs));   % prealloc 

    results_for_matrix = {};

    fprintf('Matrix: %s (%d / %d)\n', Problem.name, i, numel(matrix_files));

    for j = 1:numel(preconditioner_configs)
        config = preconditioner_configs{j};
        prec_labels(j) = prec_config2str(config);
        succeed_ichol = true; % default value

        if strcmp(config.type, "none")
            time_prec = 0;
            prec_name = "None (CG)";
            [~, result] = cg_method(A, tol=tol, max_iter=max_iter);
        else
            prec_name = config.type;

            opt = {};
            if isfield(config, 'omega')
                opt(end+1:end+2) = {'omega', config.omega};
            end
            if isfield(config, 'ictype')
                opt(end+1:end+2) = {'ictype', config.ictype};
            end
            if isfield(config, 'droptol')
                opt(end+1:end+2) = {'droptol', config.droptol};
            end


            try
                % tic for preconditioner construction
                tic;
                preconditioner = build_preconditioner(A, prec_name, opt{:});
                time_prec = toc;

                [~, result] = pcg_method(A, preconditioner, tol=tol, max_iter=max_iter);
            catch
                fprintf("FAILED: ichol\n");
                succeed_ichol = false;
                result = struct( ...
                    'iter_final', {-1}, ...
                    'is_converged', {false}, ... % boolean
                    'time', {-1}, ...
                    'hist_relres_2', {-1}, ...
                    'true_relres_2', {-1}, ...
                    'hist_relerr_2', {-1}, ...
                    'hist_relerr_A', {-1} ...
                );
            end
        end

        new_row = {Problem.name, n, prec_name, time_prec, {result}};

        results_summary = [results_summary; new_row];

        if succeed_ichol
            save_fig_prefix = strcat(extractAfter(Problem.name, "/"), "_", prec_name, "_", prec_labels(j));
            figure_title = strcat(Problem.name, "(prec:", prec_name, ", opt:", prec_labels(j), ")");
            plot_conv_hist(result, figure_title, save_fig=true, show_plot=false, ...
                save_fig_prefix=save_fig_prefix, format_type="pdf", output_dir=outputs_dir);
        end
    end

    expanded_results = expand_results_table(results_summary);
    writetable(expanded_results, strcat(outputs_dir, "/results_expanded.csv"));
end

