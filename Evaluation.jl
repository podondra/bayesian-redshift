module Evaluation

using Statistics

export rmse, cfr, computeΔv, medianΔv, meanΔv

function rmse(y, ŷ)
    return sqrt(1 / length(y) * (y - ŷ)'  * (y - ŷ))
end

function computeΔv(z_vi, z)
    # the speed of light in vacuum (km / s)
    c = 299792.458
    return c .* abs.(z - z_vi) ./ (1 .+ z_vi)
end

function cfr(y, ŷ; threshold=3000)
    return sum(computeΔv(y, ŷ) .>= threshold) / length(y)
end

function meanΔv(y, ŷ)
    return mean(computeΔv(y, ŷ))
end

function medianΔv(y, ŷ)
    return median(computeΔv(y, ŷ))
end

end # module
