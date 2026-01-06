# Set the offset
function setOffset!(offset, ν, offSetMethod)
    if offSetMethod == "kowal"
        for j = 1:size(ν,2) 
            offset[:,j] .= eps() + any(ν[:,j].^2 .< 10^-16) * 
                maximum([10^-8, mad(ν[:,j])/10^6])
        end
    end
end

# Helper function to convert log-volatility evolution to covariance matrices
LogVol2Covs(H) = PDMat.([diagm(exp.(H[t,:])) for t in 1:size(H,1)]) 

# Helper function to convert log-volatility evolution to covariance matrices
# in-place version
function LogVol2Covs!(Σᵥ, H) 
   Σᵥ .= PDMat.([diagm(exp.(H[t,:])) for t in 1:size(H,1)]) 
end