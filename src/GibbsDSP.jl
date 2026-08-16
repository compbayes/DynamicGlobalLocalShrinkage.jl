""" 
    update_dsp!(ν, S, P, H, H̃, ξ, ϕ, μ, σ²ₙ, prior, 
        mix, Dᵩ, offset = eps(), α = 1/2, β = 1/2, updateσₙ = false) 

A single Gibbs update of all dynamic shrinkage process (DSP) parameters given a T × p matrix of parameter innovations, ν. 

The columns of ν correspond to different parameters, which are assumed to be independent.
The elements in a column of ν are innovations for the log-volatility evolution, i.e.

νₜ ~ N(0, exp(hₜ/2)) for t = 1,2,...,T
hₜ = μ + ϕ(hₜ₋₁ - μ) + ηₜ, where εₜ ~ N(0,σₙ/√ξₜ)
ξₜ ~ PG(2α) is a Polya-Gamma variable.

The columns of H contains the log-volatility evolution for a given parameter: hₜ for t = 1,2,...,T. The Polya-Gamma construction gives a marginal distribution for ηₜ ~ Z(α,α,0,σₙ).

prior is a named tuple with fields ϕ₀, κ₀, m₀, σ₀, ν₀, ψ₀, which are the prior parameters for the AR coefficient ϕ, the mean μ, and the variance σ²ₙ in the log-volatility evolution.

The usual square-and-log trick in stochastic volatility models is used to turn hₜ into an additive parameter 
ỹₜ = log(νₜ² + offset) = hₜ + qₜ, where qₜ ~ log χ²₁, which is approximated by a mixture of normals distribution with mixture allocation given by the T × p matrix S. The posterior probabilities of the mixture allocation are stored in the T × p matrix P.

mix is a named tuple with fields dist, m, v, which is the mixture distribution, and the means and variances of the mixture components.

"""
function update_dsp!(ν, S, P, H, H̃, ξ, ϕ, μ, σ²ₙ, prior, mix, Dᵩ,
    offset=eps(), α=1 / 2, β=1 / 2, updateσₙ=false,
    h_upper=Inf, polyaoffset=0.0)

    p = size(ν, 2)
    Ỹ = log.(ν .^ 2 .+ offset) # Ỹ = [ỹ₁, ỹ₂,..., ỹₚ] is T × p 
    for k in 1:p

        # Update mixture allocation for log χ²₁ distribution
        S[:, k] = UpdateMixAlloc!(Ỹ[:, k], H̃[:, k] .+ μ[k], mix.dist, P)

        # Update h₁, h₂, ..., hₜ using MvNormal draw with tridiag precision matrix
        H[:, k] = Update_h(Ỹ[:, k], mix.m[S[:, k]], mix.v[S[:, k]], Dᵩ, ξ[:, k], ϕ[k],
            σ²ₙ[k], μ[k], h_upper, polyaoffset)

        # Update Polya-Gamma variables
        ξ[:, k] = Updateξ(H[:, k], ϕ[k], σ²ₙ[k], μ[k], α, β)

        # Update the AR coefficient in the logvariance evolution
        ϕ[k] = Updateϕ(H[:, k], ξ[:, k], μ[k], σ²ₙ[k], prior.ϕ₀, prior.κ₀)

        # Update the variance in the logvariance evolution
        if updateσₙ
            σ²ₙ[k] = Updateσ²ₙ(H[:, k], ξ[:, k], ϕ[k], μ[k], prior.ν₀, prior.ψ₀)
        end

        # Update the mean in the logvariance evolution - centered parameterization
        μ[k] = Updateμ(H[:, k], ξ[:, k], ϕ[k], σ²ₙ[k], prior.m₀, prior.σ₀)
        H̃[:, k] = H[:, k] .- μ[k] # Transform back to non-centered parameterization
    end

    return nothing
end

# Log conditional posterior for ϕ under grouped likelihood, used in slice sampling
function logpost_ϕ(ϕ, h, ξ, μ̄, σ²ₙ, ϕ₀, κ₀, groupsize)
    ϕ̄ = ϕ^groupsize # scalar here
    scalevar = (1 - ϕ̄^2) / (1 - ϕ^2)
    σ̄²ₙ = scalevar * σ²ₙ
    mean_h = μ̄ .+ ϕ̄ .* (h[1:end-1] .- μ̄)
    stdev_h = sqrt.(σ̄²ₙ ./ ξ[2:end])
    loglik = sum(logpdf.(Normal.(mean_h, stdev_h), h[2:end]))
    logprior = logpdf(Truncated(Normal(ϕ₀, κ₀), -1, 1), ϕ)
    return loglik + logprior
end

# Function with grouping, assuming a common groupsize.
function update_dsp!(groupsize::Int, ν, S, P, H, H̃, ξ, ϕ, μ, σ²ₙ, prior, mix, Dᵩ,
    offset=eps(), α=1 / 2, β=1 / 2, updateσₙ=false, h_upper=Inf, polyaoffset=0.0)

    p = size(ν, 2)
    Ỹ = log.(ν .^ 2 .+ offset) # Ỹ = [ỹ₁, ỹ₂,..., ỹₚ] is Tgroup × p 

    μ̄ = μ .+ log(groupsize)                             # group-level mean
    ϕ̄ = ϕ .^ groupsize                                  # group-level AR coefficient
    scalevar = @. (1 - ϕ̄^2) / (1 - ϕ^2)
    σ̄²ₙ = scalevar .* σ²ₙ      # group-level variance in Z | ξ

    for k in 1:p

        # Update mixture allocation for log χ²₁ distribution
        S[:, k] = UpdateMixAlloc!(Ỹ[:, k], H̃[:, k] .+ μ̄[k], mix.dist, P)

        # Update h₁, h₂, ..., hₜ using MvNormal draw with tridiag precision matrix
        H[:, k] = Update_h(Ỹ[:, k], mix.m[S[:, k]], mix.v[S[:, k]], Dᵩ, ξ[:, k], ϕ̄[k],
            σ̄²ₙ[k], μ̄[k], h_upper, polyaoffset)

        # Update Polya-Gamma variables # TODO: this must be generalized to unequal groups
        ξ[:, k] = Updateξ(H[:, k], ϕ̄[k], σ̄²ₙ[k], μ̄[k], α, β)

        # Update the AR coefficient in the logvariance evolution
        ϕ[k] = slice_sample_bounded(ϕ -> logpost_ϕ(ϕ, H[:, k], ξ[:, k], μ̄[k], σ²ₙ[k],
                prior.ϕ₀, prior.κ₀, groupsize), ϕ[k]; lower=-0.999, upper=0.999)

        ϕ̄ = ϕ .^ groupsize
        scalevar = @. (1 - ϕ̄^2) / (1 - ϕ^2)

        # Update the variance in the logvariance evolution
        if updateσₙ
            σ²ₙ[k] = Updateσ²ₙ(H[:, k], ξ[:, k] / scalevar[k], ϕ̄[k], μ̄[k],
                prior.ν₀, prior.ψ₀)
        end
        σ̄²ₙ = scalevar .* σ²ₙ

        # Update the mean in the logvariance evolution - centered parameterization
        μ[k] = Updateμ(H[:, k] .- log(groupsize), ξ[:, k], ϕ̄[k], σ̄²ₙ[k],
            prior.m₀, prior.σ₀)
        μ̄[k] = μ[k] + log(groupsize)

        H̃[:, k] = H[:, k] .- μ̄[k] # Transform back to non-centered parameterization
    end

    return nothing
end

