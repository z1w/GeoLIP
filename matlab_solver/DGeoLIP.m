function L = DGeoLIP(path)
    warning('off', 'all');
    weight_mats = load(path).weights;
    [~, dims] = size(weight_mats);
    weight_dims = zeros(1, dims);
    for i = 1:dims
        weight_dims(i) = size(weight_mats{i}, 2);
    end
    
    nClasses = size(weight_mats{end}, 1);
    
    n_hidden_vars = sum(weight_dims(2:end));

    %% We have shown that this compact matrix representation has a dual optimization interpretation.
    %% We reuse this representation from: https://github.com/arobey1/LipSDP/blob/master/LipSDP/matlab_engine/lipschitz_multi_layer.m
    weights = blkdiag(weight_mats{1:end-1});
    zeros_col = zeros(size(weights, 1), weight_dims(end));
    A = horzcat(weights, zeros_col);
    eyes = eye(size(A, 1));
    init_col = zeros(size(eyes, 1), weight_dims(1));
    B = horzcat(init_col, eyes);
    A_on_B = vertcat(A, B);
    extra_col = zeros(size(A_on_B, 1), 1);
    A_on_B = double(horzcat(extra_col, A_on_B));

    constraint_size = size(A_on_B, 2);

    L = zeros(nClasses, 1);
    for i = 1:nClasses
        L(i)= solve_sdp(weight_mats, i, constraint_size, weight_dims, A_on_B, n_hidden_vars);
    end 
end
