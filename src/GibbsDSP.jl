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
        offset = eps(), α = 1/2, β = 1/2, updateσₙ = false, h_upper = Inf)

    p = size(ν, 2)
    Ỹ = log.(ν.^2 .+ offset) # Ỹ = [ỹ₁, ỹ₂,..., ỹₚ] is T × p 
    for k in 1:p 

        # Update mixture allocation for log χ²₁ distribution
        S[:,k] = UpdateMixAlloc!(Ỹ[:,k], H̃[:,k] .+ μ[k], mix.dist, P)

        # Update h₁, h₂, ..., hₜ using MvNormal draw with tridiag precision matrix
        H[:,k] = Update_h(Ỹ[:,k], mix.m[S[:,k]], mix.v[S[:,k]], Dᵩ, ξ[:,k], ϕ[k], σ²ₙ[k], 
            μ[k], h_upper)

        # Update Polya-Gamma variables
        ξ[:,k] = Updateξ(H[:,k], ϕ[k], σ²ₙ[k], μ[k], α, β)

        # Update the AR coefficient in the logvariance evolution
        ϕ[k] = Updateϕ(H[:,k], ξ[:,k], μ[k], σ²ₙ[k], prior.ϕ₀, prior.κ₀)

        # Update the variance in the logvariance evolution
        if updateσₙ
            σ²ₙ[k] = Updateσ²ₙ(H[:,k], ξ[:,k], ϕ[k], μ[k], prior.ν₀, prior.ψ₀)
        end
        
        # Update the mean in the logvariance evolution - centered parameterization
        μ[k] = Updateμ(H[:,k], ξ[:,k], ϕ[k], σ²ₙ[k], prior.m₀, prior.σ₀)
        H̃[:,k] = H[:,k] .- μ[k] # Transform back to non-centered parameterization
    end

    return nothing
end