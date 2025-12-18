using DynamicalSystems, ElasticArrays

"""
    mdEmbed(data, D, τ)

Embeds a multidimensional time series into a higher-dimensional delay-embedded Dataset.

# Arguments
- `data`: Input multidimensional matrix or Dataset.
- `D`: Embedding dimension.
- `τ`: Time delay.

# Returns
- A `DynamicalSystems.Dataset` containing the embedded series.
"""

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