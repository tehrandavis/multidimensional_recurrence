
#=
This script runs step-by-step through md-RQA analysis in Julia. It uses DynamicalSystems.jl for delay embedding and analysis. 
=#

# ----- this script relies on the following packages -----
using Pkg
Pkg.add("DataFrames"); 
Pkg.add("CSV");
Pkg.add("DynamicalSystems"); # RQA stuffs
Pkg.add("Statistics");
Pkg.add("Plots"); # plotting

using DataFrames, Plots, CSV, DynamicalSystems, Statistics;

# Step 1 ----- load in an example dataset
data = DataFrame!(CSV.File("exampleData.csv"));

# turn to matrix (and in this case only use first 3 columns)
ts_names = names(data[1:3]) |> permutedims; # use this for ami plotting
data = Matrix(data[:,1:3]);

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
    plotMI[:,dimension] = mutualinformation(data[:,dimension],0:1:maxlag; nbins=nbins);
    lags[dimension] = estimate_delay(data[:,dimension], "mi_min", 0:1:maxlag; nbins=nbins);
end

τ = maximum(lags) |> round |> Int;


# Step 3 ------ find the embedding dimension
include("mdFnn.jl");

# parameters
tau = τ;
maxEmb = 10;
numSamples = 500;
Rtol = 10; 
Atol = 2;

fnn = [1:1:maxEmb];
fnnPerc = mdFnn(data[:,:], tau, maxEmb, numSamples, Rtol, Atol);

# embedding dimension
D = (findall(fnnPerc .< 1) |> minimum);

# Step 4 ------ embed timeseries
newTS = embed(data[:,:], D, τ);

# Step 5 ------ construct Recurrence Matrix
R = RecurrenceMatrix(newTS, .05; fixedrate = true);


# Step 6 ------ Recurrence quantification
rqaOUT = rqa(R; lmin = 50, thieller = 1);

# Step 7 ------- Construct output data frame

# get the obtained parameters
parameters = DataFrame(delay = tau, embed = D);

# turn rqaOUT to dataframe
rqaOUT = DataFrame([rqaOUT]);

# join parameters and rqaOUT
df_out = hcat(parameters, rqaOUT)
CSV.write("rqa_output.csv", df_out)

# optionally save to csv

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
png(diag_plots, "diagnostic_plots_filename.png")