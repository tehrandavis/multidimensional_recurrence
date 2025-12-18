
"""
This script runs step-by-step through md-RQA analysis in Julia. 
It uses DynamicalSystems.jl for delay embedding and analysis. 
"""

# ----- this script relies on the following packages -----
using Pkg
Pkg.add("DataFrames"); 
Pkg.add("CSV");
Pkg.add("DynamicalSystems"); # RQA stuffs
Pkg.add("StatsBase");
Pkg.add("Plots"); # plotting
Pkg.add("ElasticArrays"); # efficent matrix iteration
Pkg.add("LinearAlgebra") # diag caculation
Pkg.add("Distances") # matrix pairwise distances

using DataFrames, Plots, CSV, DynamicalSystems, StatsBase, ElasticArrays, LinearAlgebra, Distances;

# you should also include the follwing functions from this repo
include("mdFnn.jl")
include("mdEmbed.jl")

# Step 1 ----- load in an example dataset
data = DataFrame(CSV.File("exampleData.csv"));
#data = DataFrame(CSV.File("lorenz.csv", header=false));

# turn to matrix (and in this case only use first 3 columns)
ts_names = names(data[:,1:3]) |> permutedims; # use this for ami plotting
data = Matrix(data)[:,1:3];

# get the dimensions (number of columns) of the data
ncol = size(data,2);

# Step 2 ----- find the delay (tau)

# parameters:
nbins = 10;
maxlag = 200;

# get delay for each dimension
lags = zeros(ncol);
plotMI = zeros(maxlag+1, ncol);
for dimension in 1:ncol
    plotMI[:,dimension] = selfmutualinfo(data[:,dimension],0:1:maxlag; nbins=nbins);
    lags[dimension] = estimate_delay(data[:,dimension], "mi_min", 0:1:maxlag; nbins=nbins);
end

τ = maximum(lags) |> round |> Int;


# Step 3 ------ find the embedding dimension
# parameters
tau = τ;
maxEmb = 10;
numSamples = 500;
Rtol = 10; 
Atol = 2;
abs_threshold = 1;

fnnPerc = mdFnn(data[:,:], tau, maxEmb, numSamples, Rtol, Atol);

# embedding dimension

# simple version
D = findall(fnnPerc .< abs_threshold) |> minimum;

#= a more comple version would be to consider if the rate of FNN function is slowing. This would be useful for cases where the FNN function never goes below the threshold but does "level off"

ex:

# the minimum difference between FNN of two consecutive D's
diff_threshold = 3

D_below_absolute_threshold = findall(fnnPerc .< abs_threshold)

if size(D_below_absolute_threshold)[1]>0
    D = (findall(fnnPerc .< abs_threshold) |> minimum);
else
    # if no values are below the absolute threshold, 
    # then use the difference threshold

    D = (findall(abs.(diff(fnnPerc)) .< diff_threshold) |> minimum) + 1
end
=#

# Step 4 ------ embed timeseries
embed_TS = mdEmbed(data, D, τ);

# Step 5 ------ construct Recurrence Matrix
R = RecurrenceMatrix(embed_TS, .05; fixedrate = true);


# Step 6 ------ Recurrence quantification
rqaOUT = rqa(R; lmin = 50, thieller = 1);

# Step 7 ------- Construct output data frame

# get the obtained parameters
parameters = (delay = tau, embed = D);

# join parameters and rqaOUT
df_out = [merge(parameters, rqaOUT)] # be careful not to duplicate column names

#optionally save to csv
CSV.write("rqa_output.csv", df_out)


# Step 8 ------ Diagnostic Plots

## DO NOT ACCEPT ANY VALUES WITHOUT LOOKING AT THE PLOTS!!!!

# plot of the ami function
ami_plot = plot(plotMI[:,:], label = ts_names);
plot!(ami_plot, [τ], seriestype="vline", linestyle=:dash,label = "delay selected")

# false nearest neighbors plot
fnn_plot = plot(fnnPerc, label = "%FNN");
plot!(fnn_plot, [D], seriestype="vline", linestyle=:dash,label = "embed selected")

# recurrence plot
xs, ys = coordinates(R);
rec_plot = scatter(xs, ys, markersize = 0.1, markercolor = :black, legend = false);


# combine plots
l = @layout [a{0.5w} [b; c]]
diag_plots = plot(rec_plot, ami_plot,fnn_plot, layout = l);
plot!(diag_plots, size=(800,400))
png(diag_plots, "simple_diagnostic_plots_filename.png")