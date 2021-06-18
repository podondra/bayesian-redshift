using DLSMethod
using FITSIO
using HDF5
using LinearAlgebra
using Statistics
include("Utils.jl")
import .Utils

ε = 0.00005

@time begin
    fid = h5open("data/dr16q_superset.hdf5", "r+")
    id = read(fid["id"])
    n = size(id, 2)

    fluxes = Matrix{Float32}(undef, 3752, n)

    Threads.@threads for i = 1:n
        loglam, flux = Utils.get_spectrum("DR16Q_Superset_v3", id[:, i]...)

        # standardise flux
        flux_mean = mean(flux)
        standard_flux = Vector{Float64}((flux .- flux_mean)
                                        ./ std(flux, mean=flux_mean))

        X = Utils.polynomial_features(loglam, 3)

        n_best, a_best = dls_fit(X, standard_flux, 2, 0.9)

        idx = (Utils.LOGLAMMIN - ε .< loglam) .& (loglam .< Utils.LOGLAMMAX + ε)
        # pad with zeros
        pixels = size(flux[idx], 1)
        Δ_loglam = minimum(loglam) - Utils.LOGLAMMIN
        offset = Δ_loglam <= 0 ? 1 : round(Int, Δ_loglam * 1e4) + 1
        fluxes[offset:offset + pixels - 1, i] = (standard_flux - X * a_best)[idx]
    end

    write_dataset(fid, "X", fluxes)
    close(fid)
end
