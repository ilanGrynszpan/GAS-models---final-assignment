
using Distributions, LinearAlgebra, Random, Optim, JuMP, ForwardDiff, SpecialFunctions

### code from ChatGPT

# Define the parameter transformation
function gamma_transform(θ)
    φ, λ = θ
    α = Φ
    β = λ / φ
    return [α, β]
end

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

    α, β = gamma_transform(θ)

    # Gradient w.r.t. alpha
    grad_alpha = log(β) - digamma(α) + log(x)
    
    # Gradient w.r.t. beta
    grad_beta = α / β - x
    
    # Return as a vector
    return [grad_alpha, grad_beta]
end

### end of ChatGPT code

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


# Define the log-likelihood function and the update

## params = [Φ, k_μ, k_β, k_γ, μ_10, β_10, γ_10, γ_20, γ_30, γ_40, γ_50, γ_60, γ_70, γ_80, γ_90, γ_100, γ_110, γ_120]

Φ = 0.5
d = 0

θ = [1.0 2.0]

params = [0.5, 1.0, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
Φ, k_μ, k_β, k_γ = params[1:4]
μ_10, β_10 = params[5:6]
seasonal_init = params[7:end]

μ_t, β_t = μ_10, β_10
seasonal = seasonal_init

log_lik = 0.0
T = length(y)

for t in 1:T
    y_t = y[t]

    seasonal = seasonal_update(seasonal, t, 12, k_γ, S(y[t], d, θ)[2])

    if(t >= 2)
        β_t = β_t + k_β * (S(y[t], d, θ)[1]/S(y[t], d, θ)[2]) 
        μ_t = μ_t + β_t + k_μ * (S(y[t], d, θ)[1]/S(y[t], d, θ)[2])
    end

    month = mod(t, 12)
    if month == 0
        month = 12
    end

    seas = seasonal[month]

    println("start t = ", t)

    λ = exp(μ_t + seas)
    θ = [Φ, λ/Φ]
    println("θ = ", Φ, "   ", λ, "   ", λ/Φ)
    println(gamma_transform(θ)...)
    println(θ)
    println(seasonal[month])
    println(seasonal)
    println("gamma transf ", gamma_transform(θ))
    println("nabla ", gamma_nabla(θ, y[t]))
    println(S(y[t], d, θ)[2])
    println(μ_t)
    println(t)
    println(month)
    log_lik += logpdf(Gamma(gamma_transform(θ)...), y_t)
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