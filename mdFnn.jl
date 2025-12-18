"""
    mdFnn(data, tau, maxEmb, numSamples, Rtol, Atol)

Estimates the percentage of False Nearest Neighbors (FNN) for a multidimensional dataset.

# Arguments
- `data`: Multidimensional time series (Matrix or Dataset).
- `tau`: Time delay (integral number of samples).
- `maxEmb`: Maximum embedding dimension to test.
- `numSamples`: Number of random samples to use for FNN estimation.
- `Rtol`: Threshold for the ratio of neighbor distances in $D$ vs $D+1$ (typically 10-15).
- `Atol`: Threshold for the distance relative to the attractor size (typically 2).

# Returns
- `fnnPerc`: A Vector of FNN percentages for dimensions 1 to `maxEmb`.

Adapted from Wallot and Mønster (2018).
"""

function mdFnn(data, tau, maxEmb, numSamples, Rtol, Atol)
    if typeof(data) <: Dataset
        data = data |> Matrix
    end

    dims = size(data,2);
    N = size(data,1);

    fnnPerc = [100.0]; # first FNN

    if dims == 1
        Ra = sum(cov(data));
    else
        Ra = sum(diag(cov(data))); # estimate of attractor size
    end

    if (N-tau*(maxEmb-1)) < numSamples # check whether enough data points exist for random sampling
        numSamples = N-tau*(maxEmb-1);
        samps = 1:1:numSamples;
    else
        #numSamples = N-tau*(maxEmb-1);
        samps = sortperm(rand(numSamples));
    end
    
    embData = ElasticArray{Float64}(undef, N-tau*(maxEmb-1), 0);
    
    # Initialize variables to avoid global scope and ensure availability
    r2d = Float64[]
    yRd = Int[]

    for i = 1:maxEmb # embed data
        append!(embData,data[1+(i-1)*tau:end-(maxEmb-i)*tau, 1:dims]);
        dists =  pairwise(Euclidean(),embData.^2; dims=1);
        r2d1 = Array{Float64}(undef, 0);
        yRd1 = Array{Int}(undef, 0);
        for j = 1:numSamples # get nearest neighbors and distances
            temp = sort(dists[:,samps[j]]);
            coord = sortperm(dists[:,samps[j]]);
            push!(r2d1, temp[2]);
            push!(yRd1, coord[2] |> Int);
        end
        if i == 1
            r2d = r2d1;
            yRd = yRd1;
        else
            fnnTemp = Array{Float64}(undef, 0);
            for j = 1:length(r2d1)
                temp = dists[:,samps[j]];
                # check whether neighbors are false
                if sqrt(abs((temp[yRd[j]]^2 - r2d[j]^2)/r2d[j]^2)) > Rtol || abs(temp[yRd[j]] - r2d[j])/Ra > Atol
                    append!(fnnTemp, 1);
                else
                    append!(fnnTemp, 0);
                end
            end
            push!(fnnPerc, 100*sum(fnnTemp)/length(fnnTemp)); # compute percentage of FNN
            r2d = r2d1;
            yRd = yRd1; 
        end
    end

    return(fnnPerc)
end