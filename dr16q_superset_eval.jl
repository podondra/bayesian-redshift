### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ bdb521fa-acd6-11eb-0d41-2d68a7abecb2
using BSON, DataFrames, Flux, HDF5, Printf, Statistics, StatsBase, StatsPlots

# ╔═╡ 5c29de0c-4c3f-44ea-aea3-dbc0449a4a22
include("Evaluation.jl"); import .Evaluation

# ╔═╡ 8f22d6ac-b9f5-4437-97e0-cb5fb6006644
include("Neural.jl"); import .Neural

# ╔═╡ 62fd8899-7fe0-4d54-9326-79008c60140b
include("Utils.jl"); import .Utils

# ╔═╡ 8c971f02-0dab-41d0-928b-7937052f7542
plotly()

# ╔═╡ a96802e3-1df4-4bf7-b0d1-4e2ad106f093
Core.eval(Main, :(import Flux, NNlib))

# ╔═╡ a13527ac-4670-4a2f-a390-17700d707705
begin
	dr16q_file = h5open("data/dr16q_superset.hdf5", "r")
	X = read(dr16q_file, "X")
	id = read(dr16q_file, "id")
	df = DataFrame(
		plate=id[1, :],
		mjd=id[2, :],
		fiberid=id[3, :],
		source=read(dr16q_file, "source_z"),
		z = read(dr16q_file, "z"),
		z_10k=read(dr16q_file, "z_10k"),
		z_pca=read(dr16q_file, "z_pca"),
		z_pipe=read(dr16q_file, "z_pipe"),
		z_qn=read(dr16q_file, "z_qn"),
		z_vi=read(dr16q_file, "z_vi"))
	close(dr16q_file)
	df
end

# ╔═╡ dd19c4bb-92fd-42f6-b77b-bb0190ac793e
begin
	idx_10k = df[!, :z_10k] .> -1
	df_10k = df[idx_10k, :]
end

# ╔═╡ 1ff62a81-d0ad-44be-8f82-65487082fb25
X_10k = X[:, idx_10k]

# ╔═╡ ed949c9a-3222-4209-b8b2-1eb7383bcf93
begin
	model = BSON.load("models/mc_dropout_model.bson")[:model] |> gpu
	ẑ_10k = Neural.mc_dropout(model, X_10k |> gpu, t=20) |> cpu
end

# ╔═╡ dc462c4d-ec37-4459-87e6-87428f2229da
Evaluation.rmse(df_10k[!, :z_10k], df_10k[!, :z_pca]),
Evaluation.median_Δv(df_10k[!, :z_10k], df_10k[!, :z_pca]),
Evaluation.cat_z_ratio(df_10k[!, :z_10k], df_10k[!, :z_pca])

# ╔═╡ cf7c4ece-6bf8-4e53-85c1-1acdb2d37be1
Evaluation.rmse(df_10k[!, :z_10k], df_10k[!, :z_pipe]),
Evaluation.median_Δv(df_10k[!, :z_10k], df_10k[!, :z_pipe]),
Evaluation.cat_z_ratio(df_10k[!, :z_10k], df_10k[!, :z_pipe])

# ╔═╡ b69e11ff-1249-43fb-a423-61efa0945030
Evaluation.rmse(df_10k[!, :z_10k], df_10k[!, :z_qn]),
Evaluation.median_Δv(df_10k[!, :z_10k], df_10k[!, :z_qn]),
Evaluation.cat_z_ratio(df_10k[!, :z_10k], df_10k[!, :z_qn])

# ╔═╡ 0211a7d5-f744-4331-a55c-6860716c2109
Evaluation.rmse(df_10k[!, :z_10k], ẑ_10k),
Evaluation.median_Δv(df_10k[!, :z_10k], ẑ_10k),
Evaluation.cat_z_ratio(df_10k[!, :z_10k], ẑ_10k)

# ╔═╡ d1770126-cb47-47ae-844a-268210927dfb
begin
	@df df_10k density(:z_10k, label="Reference", xlabel="z", ylabel="Density")
	@df df_10k density!(ẑ_10k, label="MC Dropout")
end

# ╔═╡ 61f99ee5-46c0-403a-8fca-ab33f491fad8
begin
	Δv = Evaluation.compute_Δv(df_10k[!, :z], ẑ_10k)
	# random
	i = rand(1:size(df_10k, 1))
	# cat. z
	i = rand((1:size(df_10k, 1))[Δv .> 3000])
	# absolute error
	#i = sortperm(abs.(df_10k[:z] - df_10k[:ẑ]))[end]

	title = @sprintf(
		"z = %.3f; ẑ = %.2f; Δv = %.3f",
		df_10k[i, :z], ẑ_10k[i], Δv[i])
	Utils.plot_spectrum(X[:, df[!, :z_10k] .> -1][:, i], title=title, legend=:none)
	Utils.plot_spectral_lines!(df_10k[i, :z])
	Utils.plot_spectral_lines!(ẑ_10k[i], color=:red, location=:bottom)
end

# ╔═╡ eb11930b-8f4f-4301-a462-a41fa54d980f
md"## Uncertainties"

# ╔═╡ fe97db35-4c07-4c3c-9429-98665884d921
begin
	bayes = BSON.load("models/mc_dropout_model.bson")[:model] |> gpu
	trainmode!(bayes)
end

# ╔═╡ bb20eec3-c007-41ce-9881-5d956dd76c80
begin
	n = size(X_10k, 2)
	batch_size = 20
	ẑs = zeros(batch_size, n)
	entropies = zeros(n)
	mutual_infos = zeros(n)
	for idx in 1:n
		out = softmax(bayes(X_10k[:, [idx for i in 1:batch_size]] |> gpu)) |> cpu
		ẑs[:, idx] = Flux.onecold(out, 0.00f0:0.01f0:6.44f0)
		foo = mean(out, dims=2)
		entropies[idx] = -sum(foo .* log.(foo .+ eps(Float32)))
		mutual_infos[idx] = entropies[idx] + sum(
			out .* log.(out .+ eps(Float32))) / batch_size
	end
	ẑs, entropies, mutual_infos
end

# ╔═╡ 07d44911-4a18-4958-b790-ac09115af345
sortperm(mutual_infos)

# ╔═╡ 59d11a76-9ef6-4881-8d14-89ae065f3898
begin
	idx = 2530
	countmap(ẑs[:, idx])
end

# ╔═╡ 77108e12-ad9e-418c-bd02-194cb5a891c4
begin
	StatsPlots.histogram(entropies, label="Entropy")
	StatsPlots.histogram!(mutual_infos, label="Mutual Information")
end

# ╔═╡ 8928bf18-dfde-4f8b-a9dc-a70c14d52d84
begin
	ẑ_bayes = zeros(n)
	for i in 1:n
		ẑ_bayes[i] = findmax(countmap(ẑs[:, i]))[2]
	end
	ẑ_bayes
end

# ╔═╡ b6a4ead6-aa18-4aa0-be28-50e698eae60c
begin
	ts = 0.001:0.001:maximum(mutual_infos)
	y_10k = df_10k[!, :z_10k]
	plot(
		plot(
			ts,
			[sum(mutual_infos .< t) / length(mutual_infos) for t in ts],
			label="Completeness", xlabel="Threshold"),
		plot(
			ts,
			[Evaluation.cat_z_ratio(
					y_10k[mutual_infos .< t], ẑ_bayes[mutual_infos .< t])
				for t in ts],
			label="Catastrophic z Ratio", xlabel="Threshold"),
		layout=@layout [a; b])
end

# ╔═╡ 197b8de6-f7f5-4701-8e9e-220b781a0c1e
md"Due to binning, there can be a situation where the redshift is on the edge.
Therefore, the model is not sure into which bin to put its redshift.
But, we can filter it."

# ╔═╡ 94b7dc28-36d8-4a00-92fe-a7e1d65afdb0
begin
	idx_close = zeros(Bool, size(ẑs, 2))
	for k in 1:size(ẑs, 2)
		foo = sort(countmap(ẑs[:, k]))
		if length(foo) > 1
			bar = sort(collect(foo), by=x -> -x.second)[1:2]
			baz = map(x -> x.first, bar)
			idx_close[k] = abs(baz[1] - baz[2]) <= 0.015
		end
	end
end

# ╔═╡ 4c5f5b6f-8018-4ef8-91ac-6a4f47bd6afb
sum((.~idx_close) .& (mutual_infos .> 0.4))

# ╔═╡ ee0846f6-c87c-4c5f-9254-99ee70c010f8
begin
	# absolute error
	Δv_bayes = Evaluation.compute_Δv(y_10k, ẑ_bayes)
	#j = sortperm(mutual_infos)[end]
	#j = rand((1:n)[Δv_bayes .> 3000])
	#j = rand((1:n)[mutual_infos .> 0.3])
	#j = rand((1:n)[(mutual_infos .> 0.3) .& (Δv_bayes .> 3000)])
	#j = rand((1:n)[(y_10k .< 0.1) .& (ẑ_bayes .> 0.5)])
	j = rand((1:n)[(.~idx_close) .& (mutual_infos .< 0.1) .& (Δv_bayes .> 3000)])
	
	countmap(ẑs[:, j])
end

# ╔═╡ 9674fd8d-7d95-419e-95ce-8101b232e1dd
begin
	title_bayes = @sprintf(
		"z = %.3f; ẑ = %.2f; Δv = %.2f; MI = %.3f; %s",
		y_10k[j], ẑ_bayes[j], Δv_bayes[j], mutual_infos[j], df_10k[j, :source])
	Utils.plot_spectrum(X_10k[:, j], title=title_bayes, legend=:none)
	Utils.plot_spectral_lines!(y_10k[j])
	Utils.plot_spectral_lines!(ẑ_bayes[j], color=:red, location=:bottom)
	#Utils.plot_spectral_lines!(0.85, color=:red, location=:bottom)
end

# ╔═╡ b5c82204-c7ca-4c8f-a70c-ba14d35746f6
sum(idx_close)

# ╔═╡ Cell order:
# ╠═bdb521fa-acd6-11eb-0d41-2d68a7abecb2
# ╠═5c29de0c-4c3f-44ea-aea3-dbc0449a4a22
# ╠═8f22d6ac-b9f5-4437-97e0-cb5fb6006644
# ╠═62fd8899-7fe0-4d54-9326-79008c60140b
# ╠═8c971f02-0dab-41d0-928b-7937052f7542
# ╠═a96802e3-1df4-4bf7-b0d1-4e2ad106f093
# ╠═a13527ac-4670-4a2f-a390-17700d707705
# ╠═dd19c4bb-92fd-42f6-b77b-bb0190ac793e
# ╠═1ff62a81-d0ad-44be-8f82-65487082fb25
# ╠═ed949c9a-3222-4209-b8b2-1eb7383bcf93
# ╠═dc462c4d-ec37-4459-87e6-87428f2229da
# ╠═cf7c4ece-6bf8-4e53-85c1-1acdb2d37be1
# ╠═b69e11ff-1249-43fb-a423-61efa0945030
# ╠═0211a7d5-f744-4331-a55c-6860716c2109
# ╠═d1770126-cb47-47ae-844a-268210927dfb
# ╠═61f99ee5-46c0-403a-8fca-ab33f491fad8
# ╟─eb11930b-8f4f-4301-a462-a41fa54d980f
# ╠═fe97db35-4c07-4c3c-9429-98665884d921
# ╠═bb20eec3-c007-41ce-9881-5d956dd76c80
# ╠═07d44911-4a18-4958-b790-ac09115af345
# ╠═59d11a76-9ef6-4881-8d14-89ae065f3898
# ╠═77108e12-ad9e-418c-bd02-194cb5a891c4
# ╠═8928bf18-dfde-4f8b-a9dc-a70c14d52d84
# ╠═b6a4ead6-aa18-4aa0-be28-50e698eae60c
# ╠═4c5f5b6f-8018-4ef8-91ac-6a4f47bd6afb
# ╠═ee0846f6-c87c-4c5f-9254-99ee70c010f8
# ╠═9674fd8d-7d95-419e-95ce-8101b232e1dd
# ╠═197b8de6-f7f5-4701-8e9e-220b781a0c1e
# ╠═94b7dc28-36d8-4a00-92fe-a7e1d65afdb0
# ╠═b5c82204-c7ca-4c8f-a70c-ba14d35746f6
