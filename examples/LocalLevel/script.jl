# # Local level model with Dynamic Shrinkage Process parameter evolution

# In this example we explore the joint posterior in the local level model with a level following a dynamic shrinkage process prior.
#
# ```math
# \begin{align*}
#   y_t &= \theta_t + \varepsilon_t, \quad \varepsilon_t \sim N(0,\sigma_\varepsilon) \\
#   \theta_t &= \theta_{t-1} + \nu_t, \quad \nu_t \sim N\big(0,\exp(h_t/2)\big) \\
#   h_t &= \mu + \phi(h_{t-1} -\mu) + \eta_t, \quad \eta_t \sim Z(\alpha,\alpha, 0, \sigma_\eta) \\
# \end{align*}
# ```
#

# First we load the required packages and set some plotting parameters.:
using Plots, Distributions, LaTeXStrings, Random, Measures, LinearAlgebra, BandedMatrices
using ColorSchemes
using SMCsamplers, LogisticBetaDistribution
using DynamicGlobalLocalShrinkage
using Utils: quantile_multidim, ScaledInverseChiSq
using Utils: mvcolors as colors

gr(legend=:topleft, grid=false, color=colors[1], lw=1, legendfontsize=12,
    xtickfontsize=10, ytickfontsize=10, xguidefontsize=12, yguidefontsize=12,
    titlefontsize=14, markerstrokecolor=:auto)

Random.seed!(123);

# ### Simulate data from the local level model with dsp parameter evolution
T = 500;
σ²ₑ = 0.1^2;
α = 1 / 2;
ϕ = 0.8;
μ = -15;
σₙ = 1;
zdist = LogisticBeta(α, α) * σₙ
h = zeros(T + 1)
h[1] = μ # Initial value for h 
θ = zeros(T + 1)
y = zeros(T + 1)
for t in 2:T+1
    h[t] = μ + ϕ * (h[t-1] - μ) + rand(zdist)
    θ[t] = θ[t-1] + rand(Normal(0, exp(h[t] / 2)))
    y[t] = θ[t] + rand(Normal(0, sqrt(σ²ₑ)))
end
h = h[2:end];
θ = θ[2:end];
y = y[2:end];

# ### Plot the simulated data and the standard deviation
p1 = plot(y, label="time series", xlabel="time, " * L"t", ylabel=L"y_t",
    title="Simulated y", legend=:topright, color=colors[1], lw=2)
plot!(θ, label="local level", color=colors[3], lw=2)
p2 = plot(exp.(h / 2), label="standard deviation", xlabel="time, " * L"t",
    ylabel=L"\exp(h_t/2)", title="Standard deviation, " * L"\exp(h_t/2)",
    color=colors[2], lw=2);
plot(p1, p2, layout=(1, 2), size=(800, 300), xguidefontsize=12,
    yguidefontsize=12, titlefontsize=12, legend=nothing, margin=5mm, lw=2)

# ### Set up the prior, model and algorithm settings
priorSettings = (
    ϕ₀=0.5, κ₀=0.3,         # Prior for ϕ ~ N(ϕ₀, κ₀²)
    m₀=-10.0, σ₀=3.0,       # Prior for μ ~ N(m₀, σ₀²)
    ν₀=3.0, ψ₀=1.0,         # Prior for σ²ₙ ~ scaled inverse χ²(ν₀, ψ₀)
    μ₀=[0.0;;], Σ₀=[10.0;;],    # Prior for the local mean at time t=0
    νₑ=3.0, ψ²ₑ=var(y),       # Prior for noise variance σ²ₑ ~ scaled inverse χ²(νₑ, ψ²ₑ)
);
modelSettings = (
    α=1 / 2,          # First shape param in Z distribution
    β=1 / 2,          # Second shape param in Z distribution
    updateσₙ=false, # Update σ²ₙ in the Gibbs sampler, or set σₙ = 1
    nMixComp=10,    # nComp in mixture approximation of log χ²₁. Only 5 or 10 supported.
);
algoSettings = (
    nIter=10000,              # Number of iterations in the Gibbs sampler
    nBurn=3000,               # Number of burn-in iterations
    offsetMethod=eps(),       # Offset for log-volatility
);

# ### Set up the Gibbs sampler
function GibbsSamplerLocalLevelDSP(y, priorSettings, modelSettings, algoSettings)

    T = length(y)
    ϕ₀, κ₀, m₀, σ₀, ν₀, ψ₀, μ₀, Σ₀, νₑ, ψ²ₑ = priorSettings
    nIter, nBurn, offsetMethod = algoSettings
    α, β, updateσₙ, nMixComp = modelSettings

    ## Approximate the log χ²₁ distribution with a mixture of normals
    mixture = SetUpLogChi2Mixture(nMixComp) # Only 5 and 10 component supported

    ## Initial values
    p = 1                   # the number of states (only one in local level)
    S = zeros(Int, T, p)    # Mixture allocation for logχ²₁ - this is updated first
    μ = fill(m₀, p)
    if updateσₙ
        σ²ₙ = fill(ψ₀, p)
    else
        σ²ₙ = fill(1.0, p)
    end

    ϕ = fill(ϕ₀, p)
    H = fill(m₀, T, p)
    H̃ = H .- μ'
    ξ = ones(T, p)
    θ = zeros(T + 1, 1) # state evolution
    Dᵩ = BandedMatrix(-1 => repeat([-ϕ[1]], T - 1), 0 => Ones(T)) # Init D matrix for h_t
    σ²ₑ = var(y)

    ## Storage
    θpost = zeros(T + 1, nIter) # Store regression coefficients
    Hpost = zeros(T, nIter) # Store log-volatility evolution
    ϕpost = zeros(p, nIter) # Store AR coefficients
    σₙpost = zeros(p, nIter) # Store variance in log-volatility evolution
    μpost = zeros(p, nIter) # Store mean in log-volatility evolution
    σ²ₑpost = zeros(nIter) # Store measurement variance
    offset = (offsetMethod == "kowal") ? eps() * ones(T, p) : fill(offsetMethod, T, p)
    P = zeros(T, nMixComp)

    ## Set up state-space model
    A = 1.0         # State transition matrix
    B = 0.0         # Control input matrix (not used here)
    C = 1.0         # Observation matrix
    Σₑ = [σ²ₑ]        # Observation noise variance
    U = zeros(T, 1) # Control input (not used here)
    Y = y          # Observations

    for i in 1:(nBurn+nIter)

        ## Draw local level using the FFBS algorithm
        Σᵥ = LogVol2Covs(H)
        FFBS!(θ, U, Y, A, B, C, Σₑ, Σᵥ, μ₀, Σ₀)

        ## Update measurement variance
        residuals = y - θ[2:end]
        σ²ₑ = rand(ScaledInverseChiSq(νₑ + T, (νₑ * ψ²ₑ + sum(residuals .^ 2)) / (νₑ + T)))

        ## Update the log-volatility evolution
        ν = diff(θ, dims=1)
        setOffset!(offset, ν, offsetMethod)
        update_dsp!(ν, S, P, H, H̃, ξ, ϕ, μ, σ²ₙ,
            priorSettings, mixture, Dᵩ, offset, α, β, updateσₙ)

        if i > nBurn
            θpost[:, i-nBurn] .= θ
            Hpost[:, i-nBurn] .= H
            ϕpost[:, i-nBurn] = ϕ
            σₙpost[:, i-nBurn] = σ²ₙ
            μpost[:, i-nBurn] = μ
            σ²ₑpost[i-nBurn] = σ²ₑ
        end
    end

    return θpost, Hpost, ϕpost, σₙpost, μpost, σ²ₑpost

end;

# ### Run the Gibbs sampler

θpost, Hpost, ϕpost, σₙpost, μpost, σ²ₑpost = GibbsSamplerLocalLevelDSP(y,
    priorSettings, modelSettings, algoSettings);

# ### Plot the posterior distributions of the static parameters
p1 = histogram(μpost[:], title="posterior " * L"\mu", color=colors[7],
    ylabel="posterior density", normalize=true)
vline!([μ], color=:black, linestyle=:dash, label="true", lw=2)
p2 = histogram(ϕpost[:], title="posterior " * L"\phi", color=colors[7],
    ylabel="posterior density", normalize=true)
vline!([ϕ], color=:black, linestyle=:dash, label="true", lw=2)
p3 = histogram(sqrt.(σ²ₑpost), title="posterior " * L"\sigma_\varepsilon",
    color=colors[7], ylabel="posterior density", normalize=true)
vline!([sqrt(σ²ₑ)], color=:black, linestyle=:dash, label="true", lw=2)
plot(p1, p2, p3, layout=(1, 3), size=(800, 400), legend=nothing)

# ### Plot the posterior distribution of the local level evolution
θ_quantiles = quantile_multidim(θpost, [0.025, 0.5, 0.975]; dims=2)
p1 = plot(θ_quantiles[:, 2], xlabel="time",
    title="Local level evolution, " * L"\theta_t", color=colors[3], label="median",
    lw=2, legend=:bottomright)
plot!(θ_quantiles[:, 2], label="", fillrange=θ_quantiles[:, 1], lw=0,
    fillalpha=0.3, color=:gray)
plot!(θ_quantiles[:, 2], xlabel="time", label="95% C.I.",
    fillrange=θ_quantiles[:, 3], lw=0, fillalpha=0.3, color=:gray)
plot!(θ, color=colors[1], label="true", lw=2)

# ### Plot the posterior distribution of the log-volatility evolution
stdev_quantiles = quantile_multidim(exp.(Hpost / 2), [0.025, 0.5, 0.975]; dims=2)
p1 = plot(stdev_quantiles[:, 2], xlabel="time",
    title="Stdev innovations - " * L"\exp(h_t/2)", color=colors[3], label="median",
    lw=2)
plot!(stdev_quantiles[:, 2], label="", fillrange=stdev_quantiles[:, 1], lw=0,
    fillalpha=0.3, color=:gray)
plot!(stdev_quantiles[:, 2], xlabel="time", label="95% C.I.",
    fillrange=stdev_quantiles[:, 3], lw=0, fillalpha=0.3, color=:gray)
plot!(exp.(h / 2), color=colors[1], label="true", lw=2)

stdev_quantiles = quantile_multidim(Hpost, [0.025, 0.5, 0.975]; dims=2)
p2 = plot(stdev_quantiles[:, 2], xlabel="time", title=L"h_t",
    color=colors[3], label="", lw=2)
plot!(stdev_quantiles[:, 2], label="", fillrange=stdev_quantiles[:, 1], lw=0,
    fillalpha=0.3, color=:gray)
plot!(stdev_quantiles[:, 2], xlabel="time", label="",
    fillrange=stdev_quantiles[:, 3], lw=0, fillalpha=0.3, color=:gray)
plot!(h, color=colors[1], label="", lw=2)

plot(p1, p2, layout=(1, 2), size=(800, 400), legend=:topleft, margin=3mm)

