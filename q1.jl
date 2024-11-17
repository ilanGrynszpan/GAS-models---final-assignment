
## Initial setup
import Pkg
Pkg.add(["Distributions", "LinearAlgebra", "Random", "StatsBase", "StatsPlots"])
Pkg.add("Plots")
using Distributions, LinearAlgebra, Random, StatsBase, StatsPlots, Plots

Random.seed!(1234)

## Defining an AR(1) process - item iii

function run_c(T::Int64, init_val::Float64, output::String, save::Bool = true)
    Φ = 0.5
    c = 1
    σ = 1
    T = T
    y = [init_val]

    dist_ϵ = Normal(0, σ)
    ϵ = rand(dist_ϵ, T-1)

    for t in 2:T 
        new_val = c + Φ * y[t-1] + ϵ[t-1]
        push!(y,new_val)
    end

    p = plot(
        y,
        title = "Time series y, with T = $(T)",
        xlabel = "Time",
        ylabel = "y",
        label = "y"
    )

    if(save)
        savefig(output)
    end

    return y
end

run_c(100, 3.0, "output/c with T=100.png")
run_c(500, 3.0, "output/c with T=500.png")

### Getting expected value, variance and quantiles for y_{t+k}

y = []
for i in 1:5000
    push!(y, run_c(500, 3.0, "", false)[500])
end

#### Proving the expected values

Φ = 0.5
c = 1
σ = 1

expected = Φ^500 * 3.0 + (c*(1 - Φ^500))/(1 - Φ)
empirical = sum(y)/5000

println("Empirical value = $(empirical) vs. theoretical = $(expected)")

#### Variance

f = x -> (x - (Φ^500 * 3.0 + (c*(1 - Φ^500))/(1 - Φ)))^2
expected = σ^2 * (1 - Φ^(1000))/(1 - Φ^2)
empirical = sum(f.(y))/5000

println("Empirical value = $(empirical) vs. theoretical = $(expected)")

#### 0.025 qunatile

function calculate_q(quant)
    expected_ratio = quant
    variance = σ^2 * (1 - Φ^(1000))/(1 - Φ^2)
    z = quantile(Normal(0.0, 1.0), 1 - quant)
    expected_value = Φ^500 * 3.0 + (c*(1 - Φ^500))/(1 - Φ)
    q = expected_value + z * sqrt(variance)

    empirical = length(y[y .> q])/length(y)

    println("expected ratio is $(expected_ratio), empirical is $(empirical)")
end

calculate_q(0.025)
calculate_q(0.05)
calculate_q(0.1)