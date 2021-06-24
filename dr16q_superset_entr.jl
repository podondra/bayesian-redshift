using BSON
using Flux
using HDF5

include("Neural.jl")
import .Neural

const T = 20

dr16q_fid = h5open("data/dr16q_superset.hdf5", "r+")
X = read(dr16q_fid, "X") |> gpu

model = BSON.load("models/mc_dropout-wd=1e-4.bson")[:model] |> gpu
trainmode!(model)

p̄ = zeros(Neural.N_LABELS, size(X, 2))
ẑs = zeros(T, size(X, 2))

for i in 1:T
    p = softmax(Neural.forward_pass(model, X))
    ẑs[i, :] = Flux.onecold(p, Neural.LABELS) 
    global p̄ += p
end

p̄ /= T
ẑ = Flux.onecold(p̄, Neural.LABELS)
entropy = p̄ .* log.(p̄)
entropy[isnan.(entropy)] .= 0
entropy = dropdims(-sum(entropy, dims=1), dims=1)

# write to the HDF5 file
write_dataset(dr16q_fid, "z_pred", ẑ)
write_dataset(dr16q_fid, "zs_pred", ẑs)
write_dataset(dr16q_fid, "entropy", entropy)

close(dr16q_fid)
