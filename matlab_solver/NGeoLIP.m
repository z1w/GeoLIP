function lcs = NGeoLIP(path, norm)
    weight_mats = load(path).weights;
    [~, dims] = size(weight_mats);
    assert(dims==2);
    weight1 = weight_mats{1};
    weight2 = weight_mats{2};
    [hiddenSize, inSize] = size(weight1);
    outSize = size(weight2,1);
    weights = zeros(inSize, hiddenSize, outSize);
    for i = 1:inSize
        for j = 1:hiddenSize
            for k = 1:outSize
                weights(i, j, k) = weight1(j, i) * weight2(k, j);
            end
        end
    end
    lcs = zeros(outSize, 1);
    for i=1:outSize
        if norm == '2'
            lcs(i) = lip2(weights(:,:,i));
        else
            lcs(i) = lipinf(weights(:,:,i));
        end
    end
end