function opt = lip2(A)
%This implements Lipschitzness estimation given the input matrix
%This loads the two weight matrices

[~ ,n] = size(A);
M = A' * A;
e = ones(n, 1);
N = [M, M*e; e'*M, e'*M*e];

[n, ~] = size(N);
cvx_begin sdp
cvx_solver mosek
        variable X(n, n) symmetric
        maximize (trace(N * X))
        subject to
            X >= 0;
            for i = 1:n
                X(i,i) == 1;
            end
cvx_end
opt = sqrt(trace(N * X))/2;
end
