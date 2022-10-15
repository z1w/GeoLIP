function opt = lipinf(A)
    [~ ,nA] = size(A);
    en = ones(nA, 1);
    N = [A, A*en];
    [m, n] = size(N);
    M = zeros(m+n);
    for i = 1:m
        for j = 1:n
            M(i, j+m) = N(i, j);
        end
    end
    cvx_begin sdp
    cvx_solver mosek
        variable X(m+n, m+n) symmetric
        maximize (trace(M * X))
        subject to
            X >= 0;
            diag(X) == 1;
    cvx_end
    opt = trace(M * X)/2;
end