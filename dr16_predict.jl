using Flux, HDF5, Random

include("BayesianSZNet.jl")
using .BayesianSZNet

h5open("data/dr16q_superset.hdf5", "r+") do datafile
    X = gpu(read(datafile, "X"))
    model = SZNet("models/sznet.bson")
    Random.seed!(45)
    ẑs = sample(model, X)
    write_dataset(datafile, "zs_pred", ẑs)
end
