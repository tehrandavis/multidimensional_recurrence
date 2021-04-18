#=
mdFnn.jl:

a Julia based alternative for running multi-dimensional recurrence analysis. Based on Dan Mønster's code: https://github.com/danm0nster/mdembedding

requires:
using Pkg
Pkg.add("LinearAlgebra") # diag caculation
Pkg.add("Distances") # matrix pairwise distances
Pkg.add("ElasticArrays") # efficent matrices

using Distances, ElasticArrays, LinearAlgebra;
=#

function mdFnn(data, tau, maxEmb, numSamples, Rtol, Atol)
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
            global r2d = r2d1;
            global yRd = r2d1;
        else
            fnnTemp = Array{Float64}(undef, 0);
            for j = 1:length(r2d1)
                temp = dists[:,samps[j]];
                # check whether neighbors are false
                if sqrt(abs((temp[yRd1[j]] - r2d[j])/r2d[j])) > Rtol || abs(temp[yRd1[j]] - r2d[j])/Ra > Atol
                    append!(fnnTemp, 1);
                else
                    append!(fnnTemp, 0);
                end
            end
            push!(fnnPerc, 100*sum(fnnTemp)/length(fnnTemp)); # compute percentage of FNN
            r2d = r2d1;
            yRd = r2d1;
        end
    end

    return(fnnPerc)
end