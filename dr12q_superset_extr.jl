using DLSMethod
using FITSIO
using HDF5
using LinearAlgebra
using Statistics
include("Utils.jl")
import .Utils

ε = 0.00005

@time begin
    fid = h5open("data/dr12q_superset.hdf5", "r+")
    id = read(fid["id"])
    n = size(id, 2)

    fluxes = Matrix{Float32}(undef, 3752, n)

    Threads.@threads for i = 1:n
        loglam, flux = Utils.get_spectrum("Superset_DR12Q", id[:, i]...)

        # standardise flux
        flux_mean = mean(flux)
        standard_flux = Vector{Float64}((flux .- flux_mean)
                                        ./ std(flux, mean=flux_mean))

        X = Utils.polynomial_features(loglam, 3)

        n_best, a_best = dls_fit(X, standard_flux, 2, 0.9)

        idx = (Utils.LOGLAMMIN - ε .< loglam) .& (loglam .< Utils.LOGLAMMAX + ε)
        fluxes[:, i] = (standard_flux - X * a_best)[idx]
    end

    write_dataset(fid, "flux", fluxes)
    close(fid)
end
