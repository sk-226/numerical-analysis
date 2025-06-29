clc; clearvars; close all;

import utils.build_preconditioner;
import utils.pcg_method;
import utils.print_num_results;
import utils.plot_conv_hist;

% load problem
load('inputs/HB/nos7.mat');
A = sparse(Problem.A);

% SSOR
precType = 'ssor';
omega = 1.0;

n = size(A, 1);

% Start timer for preconditioner construction
tic;
preconditioner = build_preconditioner(A, precType, omega=omega);
time_prec = toc;

% Start timer for PCG method
tic;
[~, results] = pcg_method(A, preconditioner, tol=1.0e-12, max_iter=2*n);
time_pcg = toc;

% for nos7
y_lim = [1e-13 1e+4];
y_tick = [1e-12 1e-09 1e-06 1e-3 1 1e+3];
y_tick_label = {'-12', '-9', '-6', '-3', '0', '3'};

fprintf('Time for preconditioner construction: %.3f seconds\n', time_prec);
print_num_results(results, problem_name=Problem.name);
plot_conv_hist(results, problem_name=Problem.name, y_lim=y_lim, y_tick=y_tick, y_tick_label=y_tick_label);
