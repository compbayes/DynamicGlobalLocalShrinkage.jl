""" 
    update_homoscedastic_uni!(ν, H, ζ₀=4, λ²₀=exp(m₀ / 2))

Gibbs update of univariate homoscedastic variance instead of DSP
using prior σ²ₖ ~ Inv-χ²(ζ₀, λ²₀)
Still using the DSP container H, but all row are the same.
"""
function update_homoscedastic_uni!(ν, H, ζ₀=4, λ²₀=exp(m₀ / 2))
    for k = 1:size(H, 2)
        ζₙ = ζ₀ + T
        λ²ₙ = (ζ₀ * λ²₀ + sum(ν[:, k] .^ 2)) / ζₙ
        H[:, k] .= log.(rand(ScaledInverseChiSq(ζₙ, λ²ₙ)))
    end
end