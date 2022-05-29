using Flux, HDF5, Random

include("BayesianSZNet.jl")
using .BayesianSZNet

PS = ["05", "01", "005", "001"]
N_SAMPLES = 256
N_VAL = 50000
N_MODELS = 5

Random.seed!(72)
h5open("data/dr12q_superset.hdf5", "r+") do datafile
    X_va = gpu(read(datafile, "X_va"))
    ẑs = zeros(N_SAMPLES, N_VAL, N_MODELS)

    fcnn_group = create_group(datafile, "fcnn")
    sznet_group = create_group(datafile, "sznet")
    for p in PS
        for i in 1:N_MODELS
            ẑs[:, :, i] = sample(FCNN("models/fcnn_$(i)_0.$(p)_1.0e-5.bson"), X_va)
        end
        write_dataset(fcnn_group, "zs_pred_$(p)", ẑs)

        for i in 1:N_MODELS
            ẑs[:, :, i] = sample(SZNet("models/sznet_$(i)_0.$(p)_1.0e-9.bson"), X_va)
        end
        write_dataset(sznet_group, "zs_pred_$(p)", ẑs)
    end
end
