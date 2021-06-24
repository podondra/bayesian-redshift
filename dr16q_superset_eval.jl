### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ bdb521fa-acd6-11eb-0d41-2d68a7abecb2
using HDF5, Printf, Statistics, StatsBase, StatsPlots

# ╔═╡ 5c29de0c-4c3f-44ea-aea3-dbc0449a4a22
include("Evaluation.jl"); import .Evaluation

# ╔═╡ 62fd8899-7fe0-4d54-9326-79008c60140b
include("Utils.jl"); import .Utils

# ╔═╡ 8c971f02-0dab-41d0-928b-7937052f7542
plotly()

# ╔═╡ a13527ac-4670-4a2f-a390-17700d707705
begin
	dr16q_file = h5open("data/dr16q_superset.hdf5", "r")
	id = read(dr16q_file, "id")
	n = size(id, 2)
	X = read(dr16q_file, "X")
	source = read(dr16q_file, "source_z")
	z = read(dr16q_file, "z")
	z_10k = read(dr16q_file, "z_10k")
	z_pca = read(dr16q_file, "z_pca")
	z_pipe = read(dr16q_file, "z_pipe")
	z_qn = read(dr16q_file, "z_qn")
	z_vi = read(dr16q_file, "z_vi")
	z_pred = read(dr16q_file, "z_pred")
	zs_pred = read(dr16q_file, "zs_pred")
	entropy = read(dr16q_file, "entropy")
	close(dr16q_file)
end

# ╔═╡ 53e40953-7e3f-44ea-a629-7dca5d1834b1
idx_10k = z_10k .> -1

# ╔═╡ dc462c4d-ec37-4459-87e6-87428f2229da
Evaluation.rmse(z_10k[idx_10k], z_pca[idx_10k]),
Evaluation.median_Δv(z_10k[idx_10k], z_pca[idx_10k]),
Evaluation.cat_z_ratio(z_10k[idx_10k], z_pca[idx_10k])

# ╔═╡ cf7c4ece-6bf8-4e53-85c1-1acdb2d37be1
Evaluation.rmse(z_10k[idx_10k], z_pipe[idx_10k]),
Evaluation.median_Δv(z_10k[idx_10k], z_pipe[idx_10k]),
Evaluation.cat_z_ratio(z_10k[idx_10k], z_pipe[idx_10k])

# ╔═╡ b69e11ff-1249-43fb-a423-61efa0945030
Evaluation.rmse(z_10k[idx_10k], z_qn[idx_10k]),
Evaluation.median_Δv(z_10k[idx_10k], z_qn[idx_10k]),
Evaluation.cat_z_ratio(z_10k[idx_10k], z_qn[idx_10k])

# ╔═╡ 0211a7d5-f744-4331-a55c-6860716c2109
Evaluation.rmse(z_10k[idx_10k], z_pred[idx_10k]),
Evaluation.median_Δv(z_10k[idx_10k], z_pred[idx_10k]),
Evaluation.cat_z_ratio(z_10k[idx_10k], z_pred[idx_10k])

# ╔═╡ d1770126-cb47-47ae-844a-268210927dfb
begin
	idx_z = z .> -1
	density(z[idx_z], label="Reference", xlabel="z", ylabel="Density")
	density!(z_pred[idx_z], label="MC Dropout")
end

# ╔═╡ eb11930b-8f4f-4301-a462-a41fa54d980f
md"## Uncertainties"

# ╔═╡ 77108e12-ad9e-418c-bd02-194cb5a891c4
histogram(entropy, label="Entropy")

# ╔═╡ b6a4ead6-aa18-4aa0-be28-50e698eae60c
begin
	thresholds = 0.001:0.01:maximum(entropy)
	completeness = [sum(entropy .< t) / length(entropy) for t in thresholds]
	cat_zs = 100 .* [Evaluation.cat_z_ratio(z[entropy .< t], z_pred[entropy .< t])
		for t in thresholds]
	plot(
		plot(thresholds, completeness, ylabel="Completeness"),
		plot(thresholds, cat_zs,
			ylabel="Cat. z Ratio", xlabel="Threshold", legend=:none),
		layout=@layout [a; b])
end

# ╔═╡ 197b8de6-f7f5-4701-8e9e-220b781a0c1e
md"Due to binning, there can be a situation where the redshift is on the edge.
Therefore, the model is not sure into which bin to put its redshift.
But, we can filter it."

# ╔═╡ 94b7dc28-36d8-4a00-92fe-a7e1d65afdb0
begin
	idx_close = zeros(Bool, n)
	for k in 1:n
		set = collect(Set((zs_pred[:, k])))
		if length(set) == 2
			idx_close[k] = abs(set[1] - set[2]) <= 0.015
		end
	end
	sum(idx_close)
end

# ╔═╡ 3fef0ad4-6501-4780-84dd-42b4177e0eba
histogram(entropy[.~idx_close], label="Entropy")

# ╔═╡ bddc8aaa-aadd-4c2e-99ff-281fbe6b8580
Δv = Evaluation.compute_Δv(z, z_pred)

# ╔═╡ 9d29a0fb-4b8a-4ff3-bd7e-24fa96fb85d0
sum((entropy .< 1) .& (z .< 0.1) .& (z_pred .> 1))

# ╔═╡ ee0846f6-c87c-4c5f-9254-99ee70c010f8
begin
	#j = rand((1:n)[idx_close])
	#j = rand((1:n)[(entropy .> 5) .& (.~idx_close) .& (Δv .> 3000)])
	j = rand((1:n)[(entropy .< 1) .& (z .< 0.1) .& (z_pred .> 1)])
	countmap(zs_pred[:, j])
end

# ╔═╡ 9674fd8d-7d95-419e-95ce-8101b232e1dd
begin
	title = @sprintf(
		"z = %.3f; ẑ = %.2f; Δv = %.2f; E = %.3f; %s",
		z[j], z_pred[j], Δv[j], entropy[j], source[j])
	Utils.plot_spectrum(X[:, j], title=title, legend=:none)
	Utils.plot_spectral_lines!(z[j])
	Utils.plot_spectral_lines!(z_pred[j], color=:red, location=:bottom)
end

# ╔═╡ Cell order:
# ╠═bdb521fa-acd6-11eb-0d41-2d68a7abecb2
# ╠═8c971f02-0dab-41d0-928b-7937052f7542
# ╠═5c29de0c-4c3f-44ea-aea3-dbc0449a4a22
# ╠═62fd8899-7fe0-4d54-9326-79008c60140b
# ╠═a13527ac-4670-4a2f-a390-17700d707705
# ╠═53e40953-7e3f-44ea-a629-7dca5d1834b1
# ╠═dc462c4d-ec37-4459-87e6-87428f2229da
# ╠═cf7c4ece-6bf8-4e53-85c1-1acdb2d37be1
# ╠═b69e11ff-1249-43fb-a423-61efa0945030
# ╠═0211a7d5-f744-4331-a55c-6860716c2109
# ╠═d1770126-cb47-47ae-844a-268210927dfb
# ╟─eb11930b-8f4f-4301-a462-a41fa54d980f
# ╠═77108e12-ad9e-418c-bd02-194cb5a891c4
# ╠═b6a4ead6-aa18-4aa0-be28-50e698eae60c
# ╟─197b8de6-f7f5-4701-8e9e-220b781a0c1e
# ╠═94b7dc28-36d8-4a00-92fe-a7e1d65afdb0
# ╠═3fef0ad4-6501-4780-84dd-42b4177e0eba
# ╠═bddc8aaa-aadd-4c2e-99ff-281fbe6b8580
# ╠═9d29a0fb-4b8a-4ff3-bd7e-24fa96fb85d0
# ╠═ee0846f6-c87c-4c5f-9254-99ee70c010f8
# ╠═9674fd8d-7d95-419e-95ce-8101b232e1dd
