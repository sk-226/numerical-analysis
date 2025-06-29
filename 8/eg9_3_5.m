import utils.incompleteCholeskyDecomposition;

A = [4 -1 -1 0;
     -1 4 0 -1;
     -1 0 4 -1;
     0 -1 -1 4];

[L, D, deltaA] = incompleteCholeskyDecomposition(A);

display(L);
display(D);
display(deltaA);

LDLt = L * D * L';

display(LDLt);
