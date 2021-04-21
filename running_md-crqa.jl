
#=
This script runs step-by-step through md-cRQA analysis in Julia. It uses DynamicalSystems.jl for delay embedding and analysis. 
Please see 'example_embedding.jl' for more info
=#

# ----- this script relies on the following packages -----
using Pkg
Pkg.add("DataFrames"); 
Pkg.add("CSV");
Pkg.add("DelayEmbeddings");
Pkg.add("DynamicalSystems"); # RQA stuffs
Pkg.add("Statistics");
Pkg.add("Plots"); # plotting
Pkg.add("LinearAlgebra") # diag caculation
Pkg.add("Distances") # matrix pairwise distances
Pkg.add("ElasticArrays") # efficient matrices

using Distances, ElasticArrays, LinearAlgebra;
using DataFrames, CSV, DynamicalSystems, DelayEmbeddings, StatsBase, Statistics, Plots;



include("mdFnn.jl");
include("mdEmbed.jl");


# Step 1 ----- load in an example dataset
data = DataFrame(CSV.File("exampleData.csv"));

ts1 = data[1:end,2:3];
ts2 = data[1:end,5:6];

# get the dimensions (number of columns) of the data
ncol_ts1 = size(ts1,2);
ncol_ts2 = size(ts2,2);

if ncol_ts1 > 1
    ts1_names = names(ts1) |> permutedims; # use this for ami plotting
    ts1 = Matrix(ts1);
else
    ts1_names = "timeseries"
end

if ncol_ts2 > 1
    ts2_names = names(ts2) |> permutedims; # use this for ami plotting
    ts2 = Matrix(ts2);

else
    ts2_names = "timeseries"
end




# get the dimensions (number of columns) of the data
ncol_ts1 = size(ts1,2);
ncol_ts2 = size(ts2,2);

# Step 2 ----- find the delay (tau)

# parameters:
nbins = 10;
maxlag = (size(data,1) * .05) |> round |> Int;

# get delay for each dimension of ts1
lags_ts1 = zeros(ncol_ts1);

plotMI_ts1 = zeros(maxlag+1, ncol_ts1);
for dimension in 1:ncol_ts1
    plotMI_ts1[:,dimension] = mutualinformation(ts1[:,dimension],0:1:maxlag; nbins=nbins);
    lags_ts1[dimension] = estimate_delay(ts1[:,dimension], "mi_min", 0:1:maxlag; nbins=nbins);
end

# get delay for each dimension of ts2
lags_ts2 = zeros(ncol_ts2);

plotMI_ts2 = zeros(maxlag+1, ncol_ts2);
for dimension in 1:ncol_ts2
    plotMI_ts2[:,dimension] = mutualinformation(ts2[:,dimension],0:1:maxlag; nbins=nbins);
    lags_ts2[dimension] = estimate_delay(ts2[:,dimension], "mi_min", 0:1:maxlag; nbins=nbins);
end

τ = maximum([lags_ts1 lags_ts2]) |> Int;


# Step 3 ------ find the embedding dimension

# parameters
tau = τ;
maxEmb = 10;
numSamples = 500;
Rtol = 10; 
Atol = 2;
diff_threshold = 3

fnnPerc_ts1 = mdFnn(ts1, tau, maxEmb, numSamples, Rtol, Atol);
fnnPerc_ts2 = mdFnn(ts2, tau, maxEmb, numSamples, Rtol, Atol);

# embedding dimension
D_ts1_below_threshold = findall(fnnPerc_ts1 .< 1)
D_ts2_below_threshold = findall(fnnPerc_ts2 .< 1)

if size(D_ts1_below_threshold)[1]>0
    D_ts1 = (findall(fnnPerc_ts1 .< 1) |> minimum);
else
    D_ts1 = (findall(abs.(diff(fnnPerc_ts1)) .< diff_threshold) |> minimum) + 1
end

if size(D_ts2_below_threshold)[1]>0
    D_ts2 = (findall(fnnPerc_ts2 .< 1) |> minimum);
else
    D_ts2 = (findall(abs.(diff(fnnPerc_ts2)) .< diff_threshold) |> minimum) + 1
end

D = maximum([D_ts1 D_ts2]);

# Step 4 ------ embed timeseries

embed_ts1 = mdEmbed(ts1, D, τ);
embed_ts2 = mdEmbed(ts2, D, τ);

# Step 5 ------ construct Recurrence Matrix

# cross recurrence matrix
CRP = CrossRecurrenceMatrix(embed_ts1, embed_ts2, .05; fixedrate = true);

# you can create a JRP using:
# JRP = JointRecurrenceMatrix(embed_ts1, embed_ts2, .05; fixedrate = true);

# Step 6 ------ Recurrence quantification
crqaOUT = rqa(CRP; lmin = 20, thieller = 0);
#jrqaOUT = rqa(JRP; lmin = 20, thieller = 0);

# diagonal crp profile
lags = 120;
diag_RR = [];

for i = (-1*lags):1:lags # caluculate diagonal recurrences at lag i
    push!(diag_RR, 100*sum(diag(CRP,i))/length(diag(CRP,i)));
end


# Step 7 ------- Construct output data frame

# get the obtained parameters
parameters = (delay = tau, embed = D);

# join parameters and rqaOUT
df_out = [merge(parameters, rqaOUT)] # be careful not to duplicate column names

#optionally save to csv
CSV.write("crqa_output.csv", df_out)

# Step 8 ------ Diagnostic Plots

## DO NOT ACCEPT ANY VALUES WITHOUT LOOKING AT THE PLOTS!!!!

# movement timeseries plots
ts1_plot = plot(ts1, labels = ts1_names);
ts2_plot = plot(ts2, labels = ts2_names);

# plot of the ami function
ami_plot_ts1 = plot(plotMI_ts1[:,:], label = ts1_names);
plot!(ami_plot_ts1, [τ], seriestype="vline", linestyle=:dash,label = "delay selected")

ami_plot_ts2 = plot(plotMI_ts2[:,:], label = ts2_names);
plot!(ami_plot_ts2, [τ], seriestype="vline", linestyle=:dash,label = "delay selected")

# false nearest neighbors plot
fnn_plot_ts1 = plot(fnnPerc_ts1, label = "%FNN");
plot!(fnn_plot_ts1, [D], seriestype="vline", linestyle=:dash,label = "embed selected")

fnn_plot_ts2 = plot(fnnPerc_ts2, label = "%FNN");
plot!(fnn_plot_ts2, [D], seriestype="vline", linestyle=:dash,label = "embed selected")

# recurrence plot
xs, ys = coordinates(CRP);
rec_plot = scatter(xs, ys, markersize = 0.1, markercolor = :black, legend = false);

# diagonal recurrence plot
dcrp = plot(-lags:1:lags,diag_RR, legend = false); 


# combine plots
diagnostic_plots = plot(ts1_plot, ami_plot_ts1, fnn_plot_ts1, rec_plot, ts2_plot, ami_plot_ts2, fnn_plot_ts2, dcrp, layout = (2,4));
#diagnostic_plots = plot(rec_plot, ami_plot,fnn_plot, layout = l);
plot!(diagnostic_plots, size=(2000,1000));

diagnostic_plots_filename = string("cross_rec_diagnostic_plot.png")
png(diagnostic_plots, diagnostic_plots_filename)


# save output
# CSV.write("crqa_output.csv", df_out);
