function L = GeoLIP(path, norm, dual)
    warning('off', 'all');
    weight_mats = load(path).weights;
    [~, dims] = size(weight_mats);
    if dims > 2
        dual=true;
    end
    if dual
        if norm == '2'
            L = LipSDP(path);
        else
            L = DGeoLIP(path);
        end
    else
            L = NGeoLIP(path, norm);
    end
    end