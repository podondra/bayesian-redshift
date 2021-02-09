using DLSMethod
using FITSIO
using HDF5
using LinearAlgebra
using Statistics
include("Utils.jl")
import .Utils

function polynomial_features(x::Vector{Float32}, degree::Int64)::Matrix{Float64}
    X = Matrix{Float64}(undef, length(x), degree + 1)
    for col = 1:degree + 1
        X[:, col] = x .^ (col - 1)
    end
    return X
end

@time begin
    fid = h5open("data/dr16q_superset.hdf5", "r+")
    id = read(fid["id"])
    n = size(id, 1)

    fluxes = Matrix{Float32}(undef, 3826, n)

    Threads.@threads for i = 1:n
        loglam, flux = Utils.get_spectrum(id[i, :]...)

        # standardise flux
        flux_mean = mean(flux)
        standard_flux = Vector{Float64}((flux .- flux_mean)
                                        ./ std(flux, mean=flux_mean))

        X = polynomial_features(loglam, 3)

        n_best, a_best = dls_fit(X, standard_flux, 2, 0.9)

        idx = (3.58115 .<= loglam) .& (loglam .<= 3.96375)
        fluxes[:, i] = (standard_flux - X * a_best)[idx]
    end

    write_dataset(fid, "flux", fluxes)
    close(fid)
end
