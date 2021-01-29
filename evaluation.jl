module Evaluation
export rmse, catastrophic_redshift_ratio

function rmse(t, y)
    sqrt(1 / length(t) * (t - y)'  * (t - y))
end

function compute_delta_v(z, z_vi)
    # the speed of light in vacuum (km / s)
    c = 299792.458
    c .* abs.(z - z_vi) ./ (1 .+ z_vi)
end

function catastrophic_redshift_ratio(t, y)
    sum(compute_delta_v(t, y) .> 3000) / length(t)
end
end # module
