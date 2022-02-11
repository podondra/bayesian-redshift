using BSON, Flux, HDF5

include("Neural.jl")
import .Neural

dr12q = h5open("data/dr12q_superset.hdf5", "r+")
X_test = read(dr12q, "X_te")
X_test_gpu = gpu(X_test)

bayes_sznet = BSON.load("models/bayes_sznet_1e-4.bson")[:model]
bayes_sznet_gpu = gpu(bayes_sznet)
ẑ_test = Neural.mcdropout(bayes_sznet_gpu, X_test_gpu, T=20)

write_dataset(dr12q, "z_pred_te", ẑ_test)
close(dr12q)
