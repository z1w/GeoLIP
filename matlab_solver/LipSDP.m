function L = LipSDP(path)
%% This is modified from LipSDP.
%% See https://github.com/arobey1/LipSDP/blob/master/LipSDP/matlab_engine/lipschitz_multi_layer.m
    
    warning('off', 'all');
    weight_mats = load(path).weights;
    [~, dims] = size(weight_mats);
    weight_dims = zeros(1, dims);
    for i = 1:dims
        weight_dims(i) = size(weight_mats{i}, 2);
    end
    
    nClasses = size(weight_mats{end}, 1);
    
    n_hidden_vars = sum(weight_dims(2:end));

    weights = blkdiag(weight_mats{1:end-1});
    zeros_col = zeros(size(weights, 1), weight_dims(end));
    A = horzcat(weights, zeros_col);
    eyes = eye(size(A, 1));
    init_col = zeros(size(eyes, 1), weight_dims(1));
    B = horzcat(init_col, eyes);
    A_on_B = double(vertcat(A, B));
    
    constraint_size = size(A_on_B, 2);

    L = zeros(nClasses, 1);
    for i = 1:nClasses
        final_weight = weight_mats{end}(i, :);
        cvx_begin sdp
        cvx_solver mosek
            variable L_sq nonnegative
            variable D(n_hidden_vars, 1) nonnegative

            T = diag(D);
            Q = [0 * T, T; T, -2*T];

            weight_term = -1 * final_weight' * final_weight;

            M = cvx(zeros(constraint_size));
            M(1:weight_dims(1), 1:weight_dims(1)) = L_sq * eye(weight_dims(1));
            M(constraint_size-weight_dims(end)+1:end, constraint_size-weight_dims(end)+1:end) = double(weight_term);
        
            minimize L_sq
            subject to
                (A_on_B' * Q * A_on_B) - M <= 0;
        cvx_end
        sqrt(L_sq)
        L(i) = sqrt(L_sq);
    end 
end