module Evaluation

using Statistics

export rmse, cat_z_ratio, compute_Δv, median_Δv, mad_Δv

function rmse(y, ŷ)
    return sqrt(1 / length(y) * (y - ŷ)'  * (y - ŷ))
end

function compute_Δv(z_vi, z)
    # the speed of light in vacuum (km / s)
    c = 299792.458
    return c .* abs.(z - z_vi) ./ (1 .+ z_vi)
end

function cat_z_ratio(y, ŷ; threshold=3000)
    return sum(compute_Δv(y, ŷ) .>= threshold) / length(y)
end

function median_Δv(y, ŷ)
    return median(compute_Δv(y, ŷ))
end

end # module
