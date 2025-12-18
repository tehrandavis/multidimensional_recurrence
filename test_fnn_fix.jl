using DelimitedFiles
using Distances, ElasticArrays, LinearAlgebra, StatsBase

# Mock Dataset to avoid dependency on DynamicalSystems
struct Dataset end

include("mdFnn.jl")

# Load Lorenz data
data = readdlm("lorenz.csv", ',')
# Lorenz is 3D, so we expect FNN to drop to near 0 around embedding dimension 3 or 4 depending on parameters.

# Parameters similar to what might be used
tau = 15
maxEmb = 5
numSamples = 500
Rtol = 15.0
Atol = 2.0

println("Running mdFnn with:")
println("tau: $tau, maxEmb: $maxEmb, numSamples: $numSamples")

fnn_perc = mdFnn(data, tau, maxEmb, numSamples, Rtol, Atol)

println("\nFNN Percentages per dimension:")
for (i, p) in enumerate(fnn_perc)
    println("Dim $i: $p %")
end
