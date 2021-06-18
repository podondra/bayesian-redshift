### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ bdb521fa-acd6-11eb-0d41-2d68a7abecb2
begin
	using BSON, CUDA, DataFrames, Flux, HDF5, Printf, Statistics, StatsPlots
	include("Evaluation.jl"); using .Evaluation
	include("Neural.jl"); using .Neural
	include("Utils.jl"); using .Utils
	Core.eval(Main, :(import Flux, NNlib))
end

# ╔═╡ a13527ac-4670-4a2f-a390-17700d707705
begin
	dr16q_file = h5open("data/dr16q_superset.hdf5", "r")
	X = read(dr16q_file, "flux")

	model = BSON.load("models/classification_1e-4.bson")[:model] |> gpu
	ẑ = Neural.classify(model, X |> gpu) |> cpu

	id = read(dr16q_file, "id")
	df = DataFrame(
		plate=id[1, :],
		mjd=id[2, :],
		fiberid=id[3, :],
		ẑ=ẑ,
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
df_10k = df[df[:z_10k] .> -1, :]

# ╔═╡ 2b9b1d9e-7d19-47a2-b5ac-853f00a489f7
@df df_10k histogram(:z_10k)

# ╔═╡ dc462c4d-ec37-4459-87e6-87428f2229da
Evaluation.rmse(df_10k[:z_10k], df_10k[:z_pca]),
Evaluation.median_Δv(df_10k[:z_10k], df_10k[:z_pca]),
Evaluation.cat_z_ratio(df_10k[:z_10k], df_10k[:z_pca])

# ╔═╡ cf7c4ece-6bf8-4e53-85c1-1acdb2d37be1
Evaluation.rmse(df_10k[:z_10k], df_10k[:z_pipe]),
Evaluation.median_Δv(df_10k[:z_10k], df_10k[:z_pipe]),
Evaluation.cat_z_ratio(df_10k[:z_10k], df_10k[:z_pipe])

# ╔═╡ b69e11ff-1249-43fb-a423-61efa0945030
Evaluation.rmse(df_10k[:z_10k], df_10k[:z_qn]),
Evaluation.median_Δv(df_10k[:z_10k], df_10k[:z_qn]),
Evaluation.cat_z_ratio(df_10k[:z_10k], df_10k[:z_qn])

# ╔═╡ 0211a7d5-f744-4331-a55c-6860716c2109
Evaluation.rmse(df_10k[:z_10k], df_10k[:ẑ]),
Evaluation.median_Δv(df_10k[:z_10k], df_10k[:ẑ]),
Evaluation.cat_z_ratio(df_10k[:z_10k], df_10k[:ẑ])

# ╔═╡ d1770126-cb47-47ae-844a-268210927dfb
begin
	@df df_10k density(:z_10k, label="Reference", xlabel="z", ylabel="Density")
	@df df_10k density!(:ẑ, label="Clf Model")
end

# ╔═╡ 61f99ee5-46c0-403a-8fca-ab33f491fad8
begin
	Δv = Evaluation.compute_Δv(df_10k[:z], df_10k[:ẑ])
	# random
	i = rand(1:size(df_10k, 1))
	# cat. z
	i = rand((1:size(df_10k, 1))[Δv .> 3000])
	# absolute error
	#i = sortperm(abs.(df_10k[:z] - df_10k[:ẑ]))[end]

	title = @sprintf(
		"z = %.3f; ẑ = %.2f; Δv = %.3f",
		df_10k[i, :z], df_10k[i, :ẑ], Δv[i])
	Utils.plot_spectrum(X[:, df[:z_10k] .> -1][:, i], title=title, legend=:none)
	Utils.plot_spectral_lines!(df_10k[i, :z])
	prep_spec = Utils.plot_spectral_lines!(
		df_10k[i, :ẑ], color=:red, location=:bottom)
	loglam, flux = Utils.get_spectrum(
		"DR16Q_Superset_v3", df_10k[i, :plate], df_10k[i, :mjd], df_10k[i, :fiberid])
	orig_spec = plot(10 .^ loglam, flux, legend=:none)
	plot(prep_spec, orig_spec, layout=@layout [a; b])
end

# ╔═╡ eb11930b-8f4f-4301-a462-a41fa54d980f
md"## Uncertainties"

# ╔═╡ fe97db35-4c07-4c3c-9429-98665884d921
begin
	bayes = BSON.load("models/classification_1e-4.bson")[:model] |> gpu
	trainmode!(bayes)
end

# ╔═╡ bb20eec3-c007-41ce-9881-5d956dd76c80
begin
	n = size(X, 2)
	batch_size = 128
	ẑs = zeros(batch_size, n)
	entropies = zeros(n)
	mutual_infos = zeros(n)
	for idx in 1:n
		out = softmax(bayes(X[:, [idx for i in 1:batch_size]] |> gpu) |> cpu)
		ẑs[:, idx] = Flux.onecold(out, 0.00f0:0.01f0:6.44f0)
		foo = mean(out, dims=2)
		entropies[idx] = -sum(foo .* log.(foo .+ eps(Float32)))
		mutual_infos[idx] = entropies[idx] + sum(
			out .* log.(out .+ eps(Float32))) / batch_size
	end
	ẑs, entropies, mutual_infos
end

# ╔═╡ Cell order:
# ╠═bdb521fa-acd6-11eb-0d41-2d68a7abecb2
# ╠═a13527ac-4670-4a2f-a390-17700d707705
# ╠═dd19c4bb-92fd-42f6-b77b-bb0190ac793e
# ╠═2b9b1d9e-7d19-47a2-b5ac-853f00a489f7
# ╠═dc462c4d-ec37-4459-87e6-87428f2229da
# ╠═cf7c4ece-6bf8-4e53-85c1-1acdb2d37be1
# ╠═b69e11ff-1249-43fb-a423-61efa0945030
# ╠═0211a7d5-f744-4331-a55c-6860716c2109
# ╠═d1770126-cb47-47ae-844a-268210927dfb
# ╠═61f99ee5-46c0-403a-8fca-ab33f491fad8
# ╟─eb11930b-8f4f-4301-a462-a41fa54d980f
# ╠═fe97db35-4c07-4c3c-9429-98665884d921
# ╠═bb20eec3-c007-41ce-9881-5d956dd76c80
