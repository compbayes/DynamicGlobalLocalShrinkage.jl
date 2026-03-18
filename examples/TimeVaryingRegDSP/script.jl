# # Time series regression with Dynamic Shrinkage Process evolution

# In this example we explore the joint posterior in the time varying regression with a random walk evolution of the parameters following a dynamic shrinkage process prior.
#
# ```math
# \begin{align*}
#   y_t &= \boldsymbol{x}_t^\top \boldsymbol{\theta}_t + \varepsilon_t,\quad \varepsilon_t \sim N(0,\sigma_\varepsilon) \\
# \boldsymbol{\theta}_t &= \boldsymbol{\theta}_{t-1} + \boldsymbol{\nu}_t, \quad \boldsymbol{\nu}_t \sim N\Big(\boldsymbol{0},\mathrm{Diag}(\exp(\boldsymbol{h}_t/2))\Big) \\
#   \boldsymbol{h}_t &= \boldsymbol{\mu} + \phi(\boldsymbol{h}_{t-1} -\boldsymbol{\mu}) + \boldsymbol{\eta}_t, \quad \boldsymbol{\eta}_t \sim Z(\alpha,\alpha, 0, \sigma_\eta) \\
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

gr(legend = :topleft, grid = false, color = colors[1], lw = 1, legendfontsize=12,
    xtickfontsize=10, ytickfontsize=10, xguidefontsize=12, yguidefontsize=12,
    titlefontsize = 14, markerstrokecolor = :auto)

Random.seed!(123);

# ### Simulate data from the model
T = 500;
p = 2; # Number of parameters, including intercept
σ²ₑ = 0.5^2;
X = [ones(T+1) randn(T+1, p-1)]; # Design matrix
α = 1/2;
ϕ = 0.8;
μ = -15;
σₙ = 1;
zdist = fill(LogisticBeta(α, α), 2)
h = zeros(T+1,p) 
y = zeros(T+1)
θ = zeros(T+1,p) # Store the regression parameters
ϕ = fill(ϕ, p) # AR coefficient for h_t
μ = fill(μ, p) # Mean for h_t
σₙ = fill(σₙ, p) # Scale for the Z distribution
for t in 2:(T+1)
    h[t,:] = μ + diagm(ϕ)*(h[t-1,:] - μ) + rand.(σₙ.*zdist)
    θ[t,:] = θ[t-1,:] + rand.(Normal.(0, exp.(h[t,:]/2)))
    y[t] = X[t,:] ⋅ θ[t,:] + rand(Normal(0, sqrt(σ²ₑ)))
end
y = y[2:end];
X = X[2:end,:];
h = h[2:end,:]; 

# ### Plot the simulated parameter evolution of θₜ 
plt = []
for j = 1:p
    push!(plt, plot(θ[:,j], label = L"\theta_{%$(j-1)}", xlabel = "time, "*L"t", 
        ylabel = L"\theta_{%$(j-1)}", title = "Simulated "*L"\theta_{%$(j-1)}", 
        color = colors[j], lw = 2))
end
plot(plt..., layout = (p,1), size = (800, 400), xguidefontsize = 12, 
    yguidefontsize = 12, titlefontsize=12, legend = nothing, margin = 5mm)

# ### Plot the simulated parameter evolution of hₜ
plt2 = []
for j = 1:p
    push!(plt2, plot(exp.(h[:,j]/2), label = "", xlabel = "time, "*L"t", 
        title = L"\exp(h_{%$(j-1)t}/2)", 
        color = colors[j], lw = 2))
end
plot(plt2..., layout = (p,1), size = (800, 400), xguidefontsize = 12, 
    yguidefontsize = 12, titlefontsize=12, legend = nothing, margin = 5mm)    

# ### Scatter plot of the data and the regression line. 
nSnapshots = 5
timeSnapshots = round(Int,T/nSnapshots):round(Int,T/nSnapshots):T
gradcols = cgrad(:Blues, nSnapshots + 1; categorical = true, rev = false)[2:end]
xmin, xmax = extrema(X[:,2]);
pltline = scatter(X[:,2], y, marker_z = 1:T, xlabel = "x", ylabel = "y", markersize = 2,    
    color = :Blues, label = "", colorbar = false, markerstrokewidth = 0)
for (ind,t) = enumerate(timeSnapshots)
    plot!([xmin,xmax], [[1,xmin] ⋅ θ[t,:],[1,xmax] ⋅ θ[t,:]], 
        color = gradcols[ind], lw = 2, label = L"t = %$t", xlabel = "x", ylabel = "y")
end
pltline

# ### Set up the prior, model and algorithm settings
priorSettings = (
    ϕ₀ = 0.5, κ₀ = 0.3,         # Prior for ϕ ~ N(ϕ₀, κ₀²)
    m₀ = -15.0, σ₀ = 3.0,       # Prior for μ ~ N(m₀, σ₀²)
    ν₀ = 3.0, ψ₀ = 1.0,         # Prior for σ²ₙ ~ scaled inverse χ²(ν₀, ψ₀)
    μ₀ = zeros(p), Σ₀ = 10*I(p),# Prior for βₜ at time t=0
    νₑ = 3, ψ²ₑ = 1,       # Prior for noise variance σ²ₑ ~ scaled inverse χ²(νₑ, ψ²ₑ)
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
function GibbsSamplerTvRegDSP(y, X, priorSettings, modelSettings, algoSettings)

    T = length(y)
    ϕ₀, κ₀, m₀, σ₀, ν₀, ψ₀, μ₀, Σ₀, νₑ, ψ²ₑ = priorSettings
    nIter, nBurn, offsetMethod = algoSettings 
    α, β, updateσₙ, nMixComp = modelSettings

    ## Approximate the log χ²₁ distribution with a mixture of normals
    mixture = SetUpLogChi2Mixture(nMixComp) # Only 5 and 10 component supported

    ## Initial values
    p = size(X,2)           # only the errors follow a dynamic shrinkage process
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
    θ = zeros(T+1, p) # Regression coefficients evolution
    Dᵩ = BandedMatrix(-1 => repeat([-ϕ[1]], T-1), 0 => Ones(T)) # Init D matrix for h_t
    σ²ₑ = 1

    ## Storage
    θpost = zeros(T+1, p, nIter) # Store regression coefficients
    Hpost = zeros(T, p, nIter) # Store log-volatility evolution
    ϕpost = zeros(p, nIter) # Store AR coefficients
    σₙpost = zeros(p, nIter) # Store variance in log-volatility evolution
    μpost = zeros(p, nIter) # Store mean in log-volatility evolution
    σ²ₑpost = zeros(nIter) # Store measurement variance
    offset = (offsetMethod == "kowal") ? eps()*ones(T,p) : fill(offsetMethod, T, p)
    P = zeros(T, nMixComp) 

    ## Set up state-space model
    Σₑ = σ²ₑ
    A = collect(I(p))
    C = zeros(1,p,T)
    for t in 1:T
        C[:,:,t] = X[t,:]
    end
    B = 0.0
    U = zeros(T,1)
    Y = y;          # Observations

    for i in 1:(nBurn + nIter)

        ## Draw local level using the FFBS algorithm
        Σᵥ = LogVol2Covs(H)
        FFBS!(θ, U, Y, A, B, C, Σₑ, Σᵥ, μ₀, Σ₀)

        ## Update measurement variance
        residuals = [y[t] - X[t,:] ⋅ θ[t,:] for t in 1:T]
        σ²ₑ = rand(ScaledInverseChiSq(νₑ + T, (νₑ*ψ²ₑ + sum(residuals.^2))/(νₑ + T)))

        ## Update the log-volatility evolution
        ν = diff(θ, dims = 1)
        setOffset!(offset, ν, offsetMethod)
        update_dsp!(ν, S, P, H, H̃, ξ, ϕ, μ, σ²ₙ, priorSettings, mixture, Dᵩ)
        
        if i > nBurn
            θpost[:, :, i - nBurn] = θ
            Hpost[:, :, i - nBurn] = H 
            ϕpost[:, i - nBurn] = ϕ
            σₙpost[:, i - nBurn] = σ²ₙ
            μpost[:, i - nBurn] = μ
            σ²ₑpost[i - nBurn] = σ²ₑ
        end
    end
    
    return θpost, Hpost, ϕpost, σₙpost, μpost, σ²ₑpost

end;

# ### Run the Gibbs sampler

θpost, Hpost, ϕpost, σₙpost, μpost, σ²ₑpost = GibbsSamplerTvRegDSP(y, X, 
    priorSettings, modelSettings, algoSettings);

# ### Plot the posterior distributions of the static parameters
p1 = []
for j = 1:p
    push!(p1, histogram(μpost[j,:], title = "posterior "*L"\mu_{%$(j)}", 
        color = colors[7], ylabel = "posterior density", normalize = true))
end
plot(p1..., layout = (1,p), size = (800,400), legend = nothing)

p2 = []
for j = 1:p
    push!(p2, histogram(ϕpost[j,:], title = "posterior "*L"\phi_{%$(j)}", 
        color = colors[8], ylabel = "posterior density", normalize = true))
end
plot(p2..., layout = (1,p), size = (800,400), legend = nothing)

p3 = histogram(sqrt.(σ²ₑpost), title = "posterior "*L"\sigma_\varepsilon", 
    color = colors[9], ylabel = "posterior density", normalize = true)
vline!([sqrt(σ²ₑ)], color = :black, linestyle = :dash, label = "true", lw = 2)

plot(p1..., p2..., p3, layout = (3,p), size = (800,600), margin = 5mm, legend = nothing)



# ###  Plot the posterior distribution of the regression parameter evolution
plt = []
for j = 1:p
   
    stdev_quantiles =  quantile_multidim(θpost[:,j,:], [0.05, 0.5, 0.9]; dims = 2)
    p2 = plot(stdev_quantiles[:,2], xlabel = "time", title = L"\theta_{t%$(j-1)}", 
        color = colors[3], label = (j==1) ? "median" : "", lw = 2)
    plot!(stdev_quantiles[:,2], label = "", fillrange = stdev_quantiles[:,1], lw = 0, 
        fillalpha = 0.3, color =:gray)
    plot!(stdev_quantiles[:,2], xlabel = "time", label = (j==1) ? "95% C.I." : "", 
        fillrange = stdev_quantiles[:,3], lw = 0, fillalpha = 0.3, color =:gray)
    plot!(θ[:,j], color = colors[1], label = (j==1) ? "true" : "", lw = 2)

    push!(plt, p2)

end
plot(plt..., layout = (p,1), size = (800, 600), legend = :topright, margin = 5mm)

# ###  Plot the posterior distribution of the log-volatility evolution
plt = []
for j = 1:p
   
    stdev_quantiles =  quantile_multidim(Hpost[:,j,:], [0.05, 0.5, 0.9]; dims = 2)
    p2 = plot(stdev_quantiles[:,2], xlabel = "time", title = L"h_{t%$j}", 
        color = colors[3], label = "", lw = 2)
    plot!(stdev_quantiles[:,2], label = "", fillrange = stdev_quantiles[:,1], lw = 0, 
        fillalpha = 0.3, color =:gray)
    plot!(stdev_quantiles[:,2], xlabel = "time", label = "", 
        fillrange = stdev_quantiles[:,3], lw = 0, fillalpha = 0.3, color =:gray)
    plot!(h[:,j], color = colors[1], label = "", lw = 2)

    push!(plt, p2)

end
plot(plt..., layout = (p,1), size = (800, 400), legend = :topleft, margin = 5mm)