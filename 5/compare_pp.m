% Compare Gaussian elimination with and without partial pivoting
% for the system:
%   εx₁ + x₂ = 1 - δ
%   x₁ + x₂ = 1
% where ε = 1e-20 and δ = 1e-15

% Parameters
epsilon = 1e-20;
delta = 1e-15;

% System of equations:
% epsilon*x1 + x2 = 1 - delta
% x1 + x2 = 1

% Coefficient matrix and right-hand side
A = [epsilon, 1;
     1, 1];
b = [1 - delta;
     1];

% Solve using Gaussian elimination without pivoting
x_ge = gauss_elim_nopivot(A, b);

% Solve using Gaussian elimination with partial pivoting
x_gepp = gauss_elim_pp(A, b);
fprintf('  x1 = %.15e\n', x_ge(1));
fprintf('  x2 = %.15e\n\n', x_ge(2));

fprintf('Solution using Gaussian elimination with partial pivoting:\n');
fprintf('  x1 = %.15e\n', x_gepp(1));
fprintf('  x2 = %.15e\n', x_gepp(2));

% Gaussian elimination without pivoting
function x = gauss_elim_nopivot(A, b)
    n = size(A, 1);
    Ab = [A, b]; % Augmented matrix

    % Forward elimination
    for k = 1:n - 1

        for i = k + 1:n
            alpha = Ab(i, k) / Ab(k, k);
            Ab(i, k + 1:n + 1) = Ab(i, k + 1:n + 1) - alpha * Ab(k, k + 1:n + 1);
        end

    end

    % Back substitution
    x = zeros(n, 1);

    for k = n:-1:1
        x(k) = (Ab(k, n + 1) - Ab(k, k + 1:n) * x(k + 1:n)) / Ab(k, k);
    end

end

% Gaussian elimination with partial pivoting
function x = gauss_elim_pp(A, b)
    n = size(A, 1);
    Ab = [A, b]; % Augmented matrix

    % Forward elimination with partial pivoting
    for k = 1:n - 1
        % Partial pivoting
        [~, pivot] = max(abs(Ab(k:n, k)));
        pivot = pivot + k - 1;

        if pivot ~= k
            Ab([k, pivot], :) = Ab([pivot, k], :);
        end

        % Elimination
        for i = k + 1:n
            alpha = Ab(i, k) / Ab(k, k);
            Ab(i, k + 1:n + 1) = Ab(i, k + 1:n + 1) - alpha * Ab(k, k + 1:n + 1);
        end

    end

    % Back substitution
    x = zeros(n, 1);

    for k = n:-1:1
        x(k) = (Ab(k, n + 1) - Ab(k, k + 1:n) * x(k + 1:n)) / Ab(k, k);
    end

end
