function [A] = eg9_3_4_getMatrix(n)
    main_diagonal = zeros(n, 1);
    super_diagonal = -1 * ones(n-1, 1);
    sub_diagonal = -1 * ones(n-1, 1);

    for i = 1:n
        main_diagonal(i) = i;
    end

    A = diag(main_diagonal) + diag(super_diagonal, 1) + diag(sub_diagonal, -1);
end
