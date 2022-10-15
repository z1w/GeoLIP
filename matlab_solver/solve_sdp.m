function l = solve_sdp(weight_mats, i, constraint_size, weight_dims, A_on_B, n_hidden_vars)
        final_weight = weight_mats{end}(i, :);
        cvx_begin sdp
        cvx_solver mosek
            variable L_sq nonnegative
            variable D(n_hidden_vars, 1) nonnegative
            variable DualIN(weight_dims(1), 1) nonnegative
    
            T = diag(D);
            Q = [0 * T, T; T, -2*T];

            obj_term = L_sq - sum(DualIN); 
    
            M = cvx(zeros(constraint_size));
            M(1,1) = obj_term;
            M(2:weight_dims(1)+1, 2:weight_dims(1)+1) = diag(DualIN);
            M(1, constraint_size-weight_dims(end)+1:end) = double(-final_weight);
            M(constraint_size-weight_dims(end)+1:end,1) = double(-final_weight');
    
            minimize L_sq
            subject to
                (A_on_B' * Q * A_on_B) - M <= 0;
        cvx_end
        l = L_sq/2;
end