using DLSMethod
using FITSIO
using HDF5
using LinearAlgebra
using Random
using Statistics

include("BayesianSZNet.jl")
using .BayesianSZNet

ε = 0.00005
N_VAL, N_TEST = 50000, 50000

@time begin
    fid = h5open("data/dr12q_superset.hdf5", "r+")
    id = read(fid["id"])
    z_vi = read(fid["z_vi"])
    z_pipe = read(fid["z_pipe"])
    n = size(id, 2)

    fluxes = zeros(3752, n)

    Threads.@threads for i = 1:n
        loglam, flux = get_spectrum("Superset_DR12Q", id[:, i]...)

        # standardise flux
        flux_mean = mean(flux)
        standard_flux = Vector{Float64}((flux .- flux_mean) ./ std(flux, mean=flux_mean))

        X = polynomial_features(loglam, 3)
        n_best, a_best = dls_fit(X, standard_flux, 2, 0.9)

        # padding
        idx = (LOGLAMMIN - ε .< loglam) .& (loglam .< LOGLAMMAX + ε)
        pixels = sum(idx)    # count number of valid pixels
        Δ_loglam = minimum(loglam) - LOGLAMMIN
        offset = Δ_loglam <= 0 ? 1 : round(Int, Δ_loglam * 1e4) + 1
        fluxes[offset:offset + pixels - 1, i] = (standard_flux - X * a_best)[idx]
    end

    write_dataset(fid, "X", fluxes)

    # split into training, validation and test set (sizes almost according to ILSVRC)
    # seed from random.org
    Random.seed!(66)
    rnd_idx = randperm(n)
    n_tr = n - N_VAL - N_TEST
    idx_tr = rnd_idx[1:n_tr]
    idx_va = rnd_idx[n_tr + 1:n_tr + N_VAL]
    idx_te = rnd_idx[n_tr + N_VAL + 1:end]

    for (name, idx) in [("tr", idx_tr), ("va", idx_va), ("te", idx_te)]
        write_dataset(fid, "id_" * name, id[:, idx])
        write_dataset(fid, "X_" * name, fluxes[:, idx])
        write_dataset(fid, "z_vi_" * name, z_vi[idx])
        write_dataset(fid, "z_pipe_" * name, z_pipe[idx])
    end

    close(fid)
end
