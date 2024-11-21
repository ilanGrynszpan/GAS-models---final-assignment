## Initial setup
import Pkg
Pkg.add(["Distributions", "LinearAlgebra", "Random", "StatsBase", "StatsPlots", "JuMP"])
Pkg.add("Plots")
Pkg.add("Optim")
Pkg.add("ForwardDiff")
Pkg.add(["DataFrames", "CSV"])
Pkg.add("ScoreDrivenModels")
Pkg.add("SpecialFunctions")
using Distributions, LinearAlgebra, Random, StatsBase, StatsPlots, Plots, Optim, JuMP, DataFrames, CSV, ForwardDiff

Random.seed!(1234)

### Testing Gurobi

model = Model(Optim.Optimizer);
set_optimizer_attribute(model, "method", BFGS());
@variable(model, x)
@objective(model, Max, -(3.0*x^4.0 - 13.0*x^2.0 + 4.0*x + 12.0))
optimize!(model)

objective_value(model)
value.(x)

### Reading historical data into a DataFrame

df = CSV.read("./data/input/ipeadata-consumo-energia-SE.csv", DataFrame)

y = df[:,2]

### Defining the model for d=0

params = [Φ, k_μ, k_β, ]

A = [4.0 0.0; 0.0 4.0]

A^-1