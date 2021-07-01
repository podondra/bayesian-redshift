using BSON
using Flux
using HDF5
using StatsBase

include("Neural.jl")
import .Neural

const T = 20

dr16q_fid = h5open("data/dr16q_superset.hdf5", "r+")
X = read(dr16q_fid, "X") |> gpu

model = BSON.load("models/mc_dropout-wd=1e-4.bson")[:model] |> gpu
trainmode!(model)

n = size(X, 2)
p̄ = zeros(Neural.N_LABELS, n)
exp_entropy = zeros(n)
ẑs = zeros(T, n)

for i in 1:T
    p = softmax(Neural.forward_pass(model, X))
    ẑs[i, :] = Flux.onecold(p, Neural.LABELS) 
    global p̄ += p
    tmp_entropy = p .* log.(p)
    tmp_entropy[isnan.(tmp_entropy)] .= 0
    global exp_entropy += dropdims(sum(tmp_entropy, dims=1), dims=1)
end

ẑ = Flux.onecold(p̄, Neural.LABELS)

p̄ /= T
entropy = p̄ .* log.(p̄)
entropy[isnan.(entropy)] .= 0
entropy = dropdims(-sum(entropy, dims=1), dims=1)

exp_entropy /= T
mutual_information = entropy .+ exp_entropy 

variation_ratio = zeros(n)
for j in 1:n
    variation_ratio[j] = 1 - maximum(values(countmap(ẑs[:, j]))) / T
end

# write to the HDF5 file
write_dataset(dr16q_fid, "z_pred", ẑ)
write_dataset(dr16q_fid, "zs_pred", ẑs)
write_dataset(dr16q_fid, "entropy", entropy)
write_dataset(dr16q_fid, "mutual_information", mutual_information)
write_dataset(dr16q_fid, "variation_ratio", variation_ratio)

close(dr16q_fid)
