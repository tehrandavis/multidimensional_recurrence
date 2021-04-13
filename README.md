# md-recurrence-julia
multi-dimensional recurrence methods written in Julia. 

- Adapted from Wallot and Mønster (2018) "Calculation of Average Mutual Information (AMI) and False-Nearest Neighbors (FNN) for the Estimation of Embedding Parameters of Multidimensional Time Series in Matlab."
- credit to Dan Mønster: https://github.com/danm0nster/mdembedding

## Files

- **mdFnn.jl**: estimates false nearest neighbors function for multidimensional dataset

- **example_embedding.jl**: example script for running md-RQA, uses:

  - **mdFnn.jl** for multi-dimensional estimate of embedding
  - **DynamicalSystems.jl** for delay-embedding and recurrence analysis

- **exampleData.csv**: example input dataset. Contains hand position data for two participants in a joint task

- **rqa_output.csv**: example output data frame of analysis.

- **diagnostic_plots_filename.png**: output of diagnositic plot. Contains:

  - plot of AMI funciton and estimated delay (mean value of all dimensions)
  - plot of FNN function and estimated embedding
  - recurrence plot

  ![diagnostic_plots_filename](/Users/tehrandavis/Github/md-recurrence-julia/diagnostic_plots_filename.png)







