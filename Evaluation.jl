module Evaluation

export rmse, catastrophic_redshift_ratio

function rmse(t, y)
    sqrt(1 / length(t) * (t - y)'  * (t - y))
end

function compute_delta_v(z_vi, z)
    # the speed of light in vacuum (km / s)
    c = 299792.458
    c .* abs.(z - z_vi) ./ (1 .+ z_vi)
end

function catastrophic_redshift_ratio(t, y; threshold=3000)
    sum(compute_delta_v(t, y) .> threshold) / length(t)
end

end # module
