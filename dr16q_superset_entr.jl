using BSON
using Flux
using HDF5
using StatsBase

include("Neural.jl")
import .Neural

dr16q_fid = h5open("data/dr16q_superset.hdf5", "r+")
X = read(dr16q_fid, "X") |> gpu

model = BSON.load("models/classification_model.bson")[:model] |> gpu
p = softmax(Neural.forward_pass(model, X))

ẑ = Flux.onecold(p, Neural.LABELS)

entropy = p .* log.(p)
entropy[isnan.(entropy)] .= 0
entropy = dropdims(-sum(entropy, dims=1), dims=1)

# write to the HDF5 file
write_dataset(dr16q_fid, "z_pred_std", ẑ)
write_dataset(dr16q_fid, "entropy_std", entropy)

close(dr16q_fid)
