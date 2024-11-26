
using Distributions, LinearAlgebra, Random, Optim, JuMP, ForwardDiff, SpecialFunctions, CSV, DataFrames

# Define the Fisher Information matrix
function fisher_information_gamma(alpha, beta)
    # Compute the components of the Fisher Information Matrix
    I11 = trigamma(alpha)         # Trigamma function
    I12 = -1 / beta
    I21 = I12                   # Symmetry
    I22 = alpha / beta^2

    # Construct the matrix
    I = [I11 I12; I21 I22]
    return I
end

# Define the nabla vector (gradient) for the Gamma distribution
function gamma_nabla(θ, x)

    Φ, λ = θ

    println("digamma ", digamma(α), "  log(β) = ", log(β), "  y = ", x, "  log(y) = ", log(x), " α = ", α, "  β = ", β)
    # Gradient w.r.t. alpha
    grad_Φ = log(Φ) + 1 - log(λ) + log(x) - x/(λ) - digamma(Φ)
    
    # Gradient w.r.t. beta
    grad_λ = -(Φ/λ) + (Φ*x)/λ^2 
    
    # Return as a vector
    return [grad_Φ, grad_λ]
end

# Define time-varying S parameter
function S(val, d, θ)
    return fisher_information_gamma(gamma_transform(θ)...)^(-d) * gamma_nabla(θ, val)
end

# Activate time dummy
function time_dummy(t, period)
    dummies = zeros(period)
    dummies[t] = 1
    return dummies
end

# Define the seasonality update
function seasonal_update(seasonal, t, period, k_γ, S)
    new_seasonal = seasonal
    if t >= 2
        k = -k_γ/(period-1)
        for i in 1:period
            if i == T
                k = k_γ
            end
            new_seasonal[i] = seasonal[i] + k * S
        end
    end
    return new_seasonal
end


# Reading the input data

df = CSV.read("./data/input/ipeadata-consumo-energia-SE.csv", DataFrame)
y = df[:,2]

# Define the log-likelihood function and the update

## params = [Φ, k_μ, k_β, k_γ, μ_10, β_10, γ_10, γ_20, γ_30, γ_40, γ_50, γ_60, γ_70, γ_80, γ_90, γ_100, γ_110, γ_120]

Φ = 0.5
d = 0

params = [0.5, 1.0, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
Φ, k_μ, k_β, k_γ = params[1:4]
μ_10, β_10 = params[5:6]
seasonal_init = params[7:end]

μ_t, β_t = μ_10, β_10
seasonal = seasonal_init

λ_10 = exp(μ_10 + seasonal[1])
θ = [Φ, λ_10]

log_lik = 0.0
T = length(y)

for t in 1:T
    y_t = y[t]

    seasonal = seasonal_update(seasonal, t, 12, k_γ, S(y[t], d, θ)[2])

    if(t >= 2)
        β_t = β_t + k_β * S(y[t], d, θ)[2]
        μ_t = μ_t + β_t + k_μ * S(y[t], d, θ)[2]
    end

    month = mod(t, 12)
    if month == 0
        month = 12
    end

    seas = seasonal[month]

    println("start t = ", t)

    λ = exp(μ_t + seas)
    θ = [Φ, λ]
    println("θ = ", Φ, "   ", λ, "   ", λ/Φ)
    println(θ)
    println(seasonal[month])
    println(seasonal)
    println("nabla ", gamma_nabla(θ, y[t]))
    println(S(y[t], d, θ)[2])
    println(μ_t)
    println(t)
    println(month)
    log_lik += logpdf(Gamma(Φ, Φ/λ), y_t)
end


function update(params, data, d)
    Φ, k_μ, k_β, k_γ = params[1:4]
    μ_10, β_10 = params[5:6]
    seasonal_init = params[7:end]

    μ_t, β_t = zeros(2)


    T = length(data)

    log_lik = 0.0
    for t in 1:T
        y_t = data[t]
        
        if t == 1