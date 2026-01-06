using Test
using LinearAlgebra
using DynamicGlobalLocalShrinkage

@testset "DynamicGlobalLocalShrinkage.jl" begin
    T = 10; p = 4;
    Σᵥ = [zeros(p,p) for t in 1:T];
    H = zeros(T,p)
    LogVol2Covs!(Σᵥ, H)

    @test Σᵥ[1] == I(p)

    offset = eps()*ones(T,p)
    offSetMethod = "kowal"
    ν = randn(T,p)
    ν[:,1] .= eps()
    setOffset!(offset, ν, offSetMethod)
    @test all(offset[:,1] .> eps())
end
