#=
mdEmbed.jl:

a Julia based alternative for running multi-dimensional recurrence analysis. 

this function is used for multidimensional embedding. It takes a n-dimensional matrix and outputs an n*D dimensional Dataset. Important that Dataset format is state of art for DynamicalSystems functions.

example:

emb_series = mdEmbed(data, D, τ)

using DynamicalSystems
RecurrenceMatrix(emb_series, .05; fixedrate = true)

requires:
Pkg.add("ElasticArrays"); # efficent matrices
using ElasticArrays;
=#

function mdEmbed(data, D, τ)

    if typeof(data) <: Dataset
        data = data |> Matrix
    end
    
    nrows = size(data,1)
    embed_series = ElasticArray{Float64}(undef, nrows-(D-1)*τ, 0);

    for i = 1:D
        append!(embed_series,data[1+(i-1)*τ:nrows-(D-i)*τ,:])
    end
    return(Dataset(embed_series))
end