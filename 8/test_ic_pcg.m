clc; clearvars; close all;

import utils.build_preconditioner;
import utils.pcg_method;
import utils.print_num_results;
import utils.plot_conv_hist;

% load problem
load('inputs/HB/nos7.mat');
A = sparse(Problem.A);

% Incomplete Cholesky
precType = 'ic';
ictype = 'nofill';
droptol = 0.0;

n = size(A, 1);

% Start timer for preconditioner construction
tic;
preconditioner = build_preconditioner(A, precType);
time_prec = toc;

% Start timer for PCG method
tic;
[~, results] = pcg_method(A, preconditioner, tol = 1.0e-12, max_iter = 2 * n);
time_pcg = toc;

fprintf('Time for preconditioner construction: %.3f seconds\n', time_prec);
print_num_results(results, problem_name = Problem.name);
plot_conv_hist(results, problem_name = Problem.name);
