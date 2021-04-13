using Pkg
Pkg.add("CSV")
Pkg.add("DynamicalSystems")
Pkg.add("LinearAlgebra")
Pkg.add("Statistics")
Pkg.add("Gadfly")
Pkg.add("DataFrames")


using CSV, LinearAlgebra, Statistics, StatsBase, Gadfly, DataFrames;

pair = 201
trial = 1

filename = join(["Pair_", pair, "_trial_", trial])
x = DataFrame!(CSV.File(join([fn, ".csv"])))
data = Matrix(x[:,1:3])



##
nbins = 10
maxlag = 200
criterion = "firstBelow"
threshold = exp(-1)
##

lags = zeros(ncol)

for dimension in 1:ncol
    lags[dimension] = estimate_delay(data[:,dimension], "mi_min", 0:1:maxlag; nbins=10)
end

delay = (lags) |> round |> Int

a = mutualinformation(data[:,1],1:1:maxlag;nbins=10)
###

estimate_dimension(data[:3], delay, 1:10, "fnn")

plot(plotMI)
plot!([delay], seriestype="vline")
png("out.png")

