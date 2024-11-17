## Initial setup
import Pkg
Pkg.add(["Distributions", "LinearAlgebra", "Random", "StatsBase", "StatsPlots"])
Pkg.add("Plots")
using Distributions, LinearAlgebra, Random, StatsBase, StatsPlots, Plots

Random.seed!(1234)