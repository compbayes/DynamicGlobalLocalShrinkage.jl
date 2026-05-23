function update_dsp!(ν, S, H, H̃, ξ, ϕ, μ, σ²ₙ, ϕ₀, κ₀, m₀, σ₀, ν₀, ψ₀,
                     mixLogχ²₁, m, v, postDist, Dᵩ; g::Int=1, # block size (your l)
                     zprev_buf::Vector{Float64},      # length T-1
                     zcurr_buf::Vector{Float64},      # length T-1
                     prop_sd_phi::AbstractVector{<:Real},
                     acc_phi::AbstractVector{<:Integer},
                     offset=eps(),
                     updateσₙ::Bool=false,
                     α::Float64=1 / 2,
                     β::Float64=1 / 2,
                     INTERCEPT::Bool=false)
    p = size(ν, 2)
    Ỹ = similar(ν)

    @inbounds @simd for j in eachindex(ν)
        Ỹ[j] = log(ν[j]^2 + eps())
    end

    @views for k in 1:p
        # ---- grouped parameters for H|... update (conditional on current phi) ----
        phi_k = float(ϕ[k])                             # ORIGINAL phi
        phi_eff = (g == 1) ? phi_k : phi_k^g            # phi_g = phi^g
        cg_k = (g == 1) ? 1.0 : cg_sum(phi_k, g)       # c_g(phi)  (shared shock)
        sigma2_eff = (g == 1) ? float(σ²ₙ[k]) : float(σ²ₙ[k]) * (cg_k * cg_k)

        # mixture allocation
        S[:, k] = UpdateMixAlloc(Ỹ[:, k], H̃[:, k] .+ μ[k], mixLogχ²₁)

        # update H using grouped transition (conjugate conditional on phi_k)
        H[:, k] = Update_h(Ỹ[:, k], m[S[:, k]], v[S[:, k]],
                           Dᵩ, ξ[:, k], phi_eff, sigma2_eff, μ[k])
        # ---- if offset ----
        #δ = (INTERCEPT && k == 1) ? 0.00 : 0.00
        # update ξ using grouped transition quantities
        ξ[:, k] = Updateξ1(H[:, k], phi_eff, sigma2_eff, μ[k], α, β; δ=δ)

        # ---- update ORIGINAL phi via MH under grouped likelihood ----
        build_z!(zprev_buf, zcurr_buf, H[:, k], ξ[:, k], μ[k])
        ϕ_new, acc = Updateϕ_grouped_MH!(float(ϕ[k]),
                                         zprev_buf, zcurr_buf,
                                         float(σ²ₙ[k]),
                                         float(ϕ₀),
                                         float(κ₀);
                                         g=g,
                                         proposal_sd=float(prop_sd_phi[k]))

        ϕ[k] = ϕ_new
        #acc_phi[k] += acc ? 1 : 0
        # ---- optional: update σ²ₙ on ORIGINAL scale ----
        if updateσₙ
            if g == 1
                σ²ₙ[k] = Updateσ²ₙ(H[:, k], ξ[:, k], ϕ[k], μ[k], ν₀, ψ₀)
            else
                σ²ₙ[k] = Updateσ²ₙ_grouped(H[:, k], ξ[:, k], phi_eff, cg_k, μ[k], ν₀, ψ₀)
            end
        end

        # update μ 
        #μ[k] = Updateμ(H[:,k], ξ[:,k], ϕ[k], σ²ₙ[k], m₀, σ₀)
        μ[k] = Updateμ(H[:, k], ξ[:, k], phi_eff, sigma2_eff, m₀, σ₀)
        @. H̃[:, k] = H[:, k] - μ[k]
    end
    return nothing
end

# --- MH update for original phi under grouped likelihood ---

function Updateϕ_grouped_MH!(phi::Float64,
                             # current value of AR coefficient ϕ
                             zprev::Vector{Float64},
                             # lagged states/observations z_{t-1} (buffer)
                             zcurr::Vector{Float64},
                             # current states/observations z_t (buffer)
                             sigma2::Float64,              # innovation variance σ²
                             phi0::Float64,                # prior mean for ϕ
                             kappa0::Float64;              # prior std for ϕ
                             g::Int=1,
                             # grouping size used in grouped likelihood
                             proposal_sd::Float64=0.05
                             # RW-MH proposal standard deviation (too small?)
                             )
    # --- log prior for phi --- Enforces |ϕ| < 1.

    @inline function logprior(phi::Float64)
        return (abs(phi) < 1.0) ? logpdf(Normal(phi0, kappa0), phi) : -Inf
    end

    # --- current log posterior ---
    # Posterior = prior + grouped likelihood
    lp_cur = logprior(phi) + loglik_phi_grouped(zprev, zcurr, phi, g, sigma2)

    if !isfinite(lp_cur)
        # Try drawing feasible values from the truncated prior
        for _ in 1:50
            # Draw proposal restricted to stationary region
            cand = rand(Truncated(Normal(phi0, kappa0), -1, 1))
            # Evaluate posterior at candidate
            lp = logprior(cand) + loglik_phi_grouped(zprev, zcurr, cand, g, sigma2)
            # Return first numerically valid candidate
            if isfinite(lp)
                return cand, true
            end
        end
        return phi,
               false
    end

    # --- random walk proposal ---
    #   ϕ* = ϕ + ε,   ε ~ N(0, proposal_sd²)
    phi_prop = phi + proposal_sd * randn()

    # --- proposed log posterior ---

    lp_prop = logprior(phi_prop) + loglik_phi_grouped(zprev, zcurr, phi_prop, g, sigma2)

    # --- Metropolis-Hastings acceptance step ---
    #   α = min(1, exp(lp_prop - lp_cur))
    if log(rand()) < (lp_prop - lp_cur)
        # accept proposal
        return phi_prop,
               true
    else
        # reject proposal
        return phi,
               false
    end
end

# --- z buffers for MH step ---

function build_z!(zprev::Vector{Float64}, zcurr::Vector{Float64},
                  H::AbstractVector{<:Real}, ξ::AbstractVector{<:Real}, μ::Real)
    T = length(H)
    @inbounds for t in 2:T
        s = sqrt(float(ξ[t]))
        zprev[t - 1] = s * (float(H[t - 1]) - μ)
        zcurr[t - 1] = s * (float(H[t]) - μ)
    end
    return nothing
end

# --- helper: stable c_g = 1 + ϕ + ... + ϕ^(g-1) (avoids 0/0 near ϕ≈1) ---
@inline function cg_sum(phi::Float64, g::Int)
    s = 1.0
    p = 1.0
    @inbounds for _ in 2:g
        p *= phi
        s += p
    end
    return s
end