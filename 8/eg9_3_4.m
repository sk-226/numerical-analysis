clc; clearvars; close all;

import utils.eg9_3_4_getMatrix;
import utils.cg_method;
import utils.print_num_results;
import utils.plot_conv_hist;

n = 5000; % size of the matrix used in the example 9.3.4
A = sparse(eg9_3_4_getMatrix(n));

% perform CG method
[~, results] = cg_method(A, tol=1.0e-10, max_iter=2*n);

print_num_results(results, problem_name='Example 9.3.4');
plot_conv_hist(results, problem_name='Example 9.3.4');
