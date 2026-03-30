# # Time series regression with Dynamic Shrinkage Process Stochastic Volatility

# In this example we explore the joint posterior in the time series regression with stochastic volatility model for the error following a dynamic shrinkage process prior.
#
# ```math
# \begin{align*}
#   y_t &= \boldsymbol{x}_t^\top \boldsymbol{\beta} + \nu_t, \quad \nu_t \sim N\big(0,\exp(h_t)\big) \\
#   h_t &= \mu + \phi(h_{t-1} -\mu) + \eta_t, \quad \eta_t \sim Z(\alpha,\alpha, 0, \sigma_\eta) \\
# \end{align*}
# ```
#

# First we load the required packages and set some plotting parameters.:
using Plots, Distributions, LaTeXStrings, Random, Measures, LinearAlgebra, BandedMatrices
using ColorSchemes
using LogisticBetaDistribution
using DynamicGlobalLocalShrinkage
using Utils: quantile_multidim, ScaledInverseChiSq
using Utils: mvcolors as colors

gr(legend = :topleft, grid = false, color = colors[1], lw = 1, legendfontsize=12,
    xtickfontsize=10, ytickfontsize=10, xguidefontsize=12, yguidefontsize=12,
    titlefontsize = 14, markerstrokecolor = :auto)

Random.seed!(123);

# ### Simulate data from the model
T = 1000;
βvect = [1,-1];
X = [ones(T+1) randn(T+1, length(βvect)-1)]; # Design matrix
α = 1/2;
ϕ = 0.8;
μ = -4;
σₙ = 1;
zdist = LogisticBeta(α, α)*σₙ
h = zeros(T+1) 
y = zeros(T+1)
for t in 2:T+1
    h[t] = μ + ϕ*(h[t-1] - μ) + rand(zdist)
    y[t] = X[t,:] ⋅ βvect + rand(Normal(0, exp(h[t]/2)))
end
y = y[2:end];
X = X[2:end,:];
h = h[2:end]; 

# ### Plot the simulated data and the standard deviation
p1 = plot(y, label = "time series", xlabel = "time, "*L"t", ylabel = L"y_t", 
    title = "Simulated y", legend = :topright, color = colors[1], lw = 2)
p2 = plot(exp.(h/2), label = "standard deviation", xlabel = "time, "*L"t", 
    ylabel = L"\exp(h_t/2)", title = "Standard deviation, "*L"\exp(h_t/2)", 
    color = colors[2], lw = 2);
plot(p1, p2, layout = (1,2), size = (800, 300), xguidefontsize = 12, 
    yguidefontsize = 12, titlefontsize=12, legend = nothing, margin = 5mm, lw = 2);
plot(p1, p2, layout = (1,2), size = (800, 300), xguidefontsize = 12, 
    yguidefontsize = 12, titlefontsize=12, legend = nothing, margin = 5mm)

# Scatter plot of the data and the regression line. The color of the markers indicate the time of the observation.
xmin, xmax = extrema(X[:,2]);
p3 = plot([xmin,xmax], [[1,xmin] ⋅ βvect,[1,xmax] ⋅ βvect], color = :black, lw = 2,
    label = "", xlabel = "x", ylabel = "y", title = "Color indicates time");
scatter!(X[:,2], y, markersize = 4, marker_z = 1:T, color = :Blues, markerstrokewidth = 0,
    label = "", xlabel = "x", ylabel = "y", title = "Darker color is later in time", colorbar = false);
p3

# ### Set up the prior, model and algorithm settings
priorSettings = (
    ϕ₀ = 0.5, κ₀ = 0.3,     # Prior for ϕ ~ N(ϕ₀, κ₀²)
    m₀ = -5.0, σ₀ = 3.0,    # Prior for μ ~ N(m₀, σ₀²)
    ν₀ = 3.0, ψ₀ = 1.0,     # Prior for σ²ₙ ~ scaled inverse χ²(ν₀, ψ₀)
); 
modelSettings = (
    α = 1/2,          # First shape param in Z distribution
    β = 1/2,          # Second shape param in Z distribution
    updateσₙ = false, # Update σ²ₙ in the Gibbs sampler, or set σₙ = 1
    nMixComp = 10,    # nComp in mixture approximation of log χ²₁. Only 5 or 10 supported.
);
algoSettings = (
    nIter = 10000,              # Number of iterations in the Gibbs sampler
    nBurn = 3000,               # Number of burn-in iterations
    offsetMethod = eps(),       # Offset for log-volatility
);

# ### Set up the Gibbs sampler
function GibbsSamplerRegressionDSPErrors(y, X, priorSettings, modelSettings, algoSettings)

    T = length(y)
    ϕ₀, κ₀, m₀, σ₀, ν₀, ψ₀ = priorSettings
    nIter, nBurn, offsetMethod = algoSettings 
    α, β, updateσₙ, nMixComp = modelSettings

    ## Approximate the log χ²₁ distribution with a mixture of normals
    mixture = SetUpLogChi2Mixture(nMixComp) # Only 5 and 10 component supported

    ## Initial values
    p = 1                   # only the errors follow a dynamic shrinkage process
    q = size(X, 2)          # Number of β coefficients
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
    βreg = zeros(p) # Regression coefficients
    Dᵩ = BandedMatrix(-1 => repeat([-ϕ[1]], T-1), 0 => Ones(T)) # Init D matrix for h_t

    ## Storage
    βpost = zeros(q, nIter) # Store regression coefficients
    Hpost = zeros(T, nIter) # Store log-volatility evolution
    ϕpost = zeros(p, nIter) # Store AR coefficients
    σₙpost = zeros(p, nIter) # Store variance in log-volatility evolution
    μpost = zeros(p, nIter) # Store mean in log-volatility evolution
    offset = (offsetMethod == "kowal") ? eps()*ones(T,p) : fill(offsetMethod, T, p) 
    P = zeros(T, nMixComp) # Set up storage for posterior of the mixture allocation
    for i in 1:(nBurn + nIter)

        ## Draw regression coefficients using Bayesian WLS (uniform prior)
        d = 1 ./ exp.(H[:,1]/2) # 1/std for pre-whitening
        ỹ = d .* y
        X̃ = repeat(d, 1, q) .* X
        βreg = rand(MvNormalCanon(X̃'*ỹ, X̃'*X̃)) # Variance of ε̃ is 1 after pre-whitening

        ## Update the log-volatility evolution
        ν = y - X*βreg
        setOffset!(offset, ν, offsetMethod)
        update_dsp!(ν, S, P, H, H̃, ξ, ϕ, μ, σ²ₙ, priorSettings, mixture, Dᵩ)
        
        if i > nBurn
            βpost[:, i - nBurn] = βreg
            Hpost[:, i - nBurn] = H[:, 1] # Only one parameter in this case
            ϕpost[:, i - nBurn] = ϕ
            σₙpost[:, i - nBurn] = σ²ₙ
            μpost[:, i - nBurn] = μ
        end
    end
    
    return βpost, Hpost, ϕpost, σₙpost, μpost

end;

# ### Run the Gibbs sampler

βpost, Hpost, ϕpost, σₙpost, μpost = GibbsSamplerRegressionDSPErrors(y, X, 
    priorSettings, modelSettings, algoSettings);

# ### Plot the posterior distributions of the static parameters
p1 = histogram(βpost[1,:], title = "posterior "*L"\beta_0", color = colors[7],
    ylabel = "posterior density", normalize = true)
vline!([βvect[1]], color = :black, linestyle = :dash, label = "true", lw = 2)
p2 = histogram(βpost[2,:], title = "posterior "*L"\beta_1", color = colors[7], 
    ylabel = "posterior density", normalize = true)
vline!([βvect[2]], color = :black, linestyle = :dash, label = "true", lw = 2)
p3 = histogram(μpost[:], title = "posterior "*L"\mu", color = colors[7], 
    ylabel = "posterior density", normalize = true)
vline!([μ], color = :black, linestyle = :dash, label = "true", lw = 2)
p4 = histogram(ϕpost[:], title = "posterior "*L"\phi", color = colors[7], 
ylabel = "posterior density", normalize = true)
vline!([ϕ], color = :black, linestyle = :dash, label = "true", lw = 2)
plot(p1, p2, p3, p4, layout = (2,2), size = (800,400), legend = nothing)

# ###  Plot the posterior distribution of the log-volatility evolution
stdev_quantiles =  quantile_multidim(exp.(Hpost/2), [0.025, 0.5, 0.975]; dims = 2)
p1 = plot(stdev_quantiles[:,2], xlabel = "time", 
    title = "Stdev innovations - "*L"\exp(h_t/2)", color = colors[3], label = "median", 
    lw = 2)
plot!(stdev_quantiles[:,2], label = "", fillrange = stdev_quantiles[:,1], lw = 0, 
    fillalpha = 0.3, color =:gray)
plot!(stdev_quantiles[:,2], xlabel = "time", label = "95% C.I.", 
    fillrange = stdev_quantiles[:,3], lw = 0, fillalpha = 0.3, color =:gray)
plot!(exp.(h/2), color = colors[1], label = "true", lw = 2)

stdev_quantiles =  quantile_multidim(Hpost, [0.025, 0.5, 0.975]; dims = 2)
p2 = plot(stdev_quantiles[:,2], xlabel = "time", title = L"h_t", 
    color = colors[3], label = "", lw = 2)
plot!(stdev_quantiles[:,2], label = "", fillrange = stdev_quantiles[:,1], lw = 0, 
    fillalpha = 0.3, color =:gray)
plot!(stdev_quantiles[:,2], xlabel = "time", label = "", 
    fillrange = stdev_quantiles[:,3], lw = 0, fillalpha = 0.3, color =:gray)
plot!(h, color = colors[1], label = "", lw = 2)

plot(p1, p2, layout = (1,2), size = (800,400), legend = :topleft, margin = 3mm)
