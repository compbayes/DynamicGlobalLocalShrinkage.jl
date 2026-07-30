""" 
    update_homoscedastic_uni!(ν, H, ζ₀, λ²₀)

Gibbs update of univariate homoscedastic variance instead of DSP
using prior σ²ₖ ~ Inv-χ²(ζ₀, λ²₀)
Still using the DSP container H, but all row are the same.
"""
function update_homoscedastic_uni!(ν, H, ζ₀, λ²₀)
    T, p = size(ν)
    for k = 1:p
        ζₙ = ζ₀ + T
        λ²ₙ = (ζ₀ * λ²₀ + sum(ν[:, k] .^ 2)) / ζₙ
        H[:, k] .= log.(rand(ScaledInverseChiSq(ζₙ, λ²ₙ)))
    end
end