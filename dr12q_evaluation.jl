### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# ╔═╡ 6dc764b2-66e7-11eb-0833-9dc54a18f920
using BSON, CUDA, FITSIO, Flux, HDF5, Printf, Statistics, StatsBase, StatsPlots

# ╔═╡ 21d5a8d1-d087-4598-851f-8cb8e67fee83
begin
	include("Evaluation.jl"); using .Evaluation
	include("Neural.jl"); using .Neural
	include("Utils.jl"); using .Utils
end

# ╔═╡ ed4d438e-6aaa-11eb-051e-efe644cce631
md"# Evaluation"

# ╔═╡ 8643602a-66e9-11eb-3700-374047551428
begin
	dr12q_file = h5open("data/dr12q_superset.hdf5", "r")
	id = read(dr12q_file, "id_va")
	X = read(dr12q_file, "X_va") |> gpu
	y = read(dr12q_file, "z_vi_va")
	y_pipe = read(dr12q_file, "z_pipe_va")
	close(dr12q_file)
	size(id), size(X), size(y), typeof(X), typeof(y)
end

# ╔═╡ 646c8844-d25a-4453-a4d9-e6f6279c183b
md"## Regression Model"

# ╔═╡ edd6e898-6797-11eb-2cee-791764fb425a
begin
	Core.eval(Main, :(import Flux, NNlib))
	regression_model = BSON.load("models/regression_model.bson")[:model] |> gpu
	ŷ_regression = Neural.regress(regression_model, X) |> cpu
	regression_model
end

# ╔═╡ 7b0e34c6-67ba-11eb-2500-378603362df8
Evaluation.rmse(y, ŷ_regression),
Evaluation.median_Δv(y, ŷ_regression),
Evaluation.mad_Δv(y, ŷ_regression),
Evaluation.cat_z_ratio(y, ŷ_regression)

# ╔═╡ e6d5d6fa-6aab-11eb-0434-19c6a5a97099
begin
	density(y, label="Visual z", xlabel="z", ylabel="Density")
	density!(ŷ_regression, label="Regression z")
end

# ╔═╡ 5f25ed6f-ef8d-49c0-993d-e3d5c445fd99
scatter(y, ŷ_regression, legend=:none, xlabel="Visual", ylabel="Regression")

# ╔═╡ 1d52576a-6abb-11eb-2dd8-c388b2365ddd
begin	
	Δv_regression = Evaluation.compute_Δv(y, ŷ_regression)
	# random
	i = rand(1:size(id, 2))
	# cat. z
	i = rand((1:size(id, 2))[Δv_regression .> 3000])
	# absolute error
	#i = sortperm(abs.(y - ŷ_regression))[end]

	Utils.plot_spectrum(
		X[:, i], legend=:none,
		title=@sprintf(
			"z = %f; ẑ = %f; Δv = %f",
			y[i], ŷ_regression[i], Δv_regression[i]))
	Utils.plot_spectral_lines!(y[i])
	regression_plot = Utils.plot_spectral_lines!(
		ŷ_regression[i], color=:red, location=:bottom)
	plot(
		regression_plot,
		plot(Utils.get_linear_spectrum(
				"Superset_DR12Q", id[:, i]...)..., legend=:none),
		layout=@layout [a; b])
end

# ╔═╡ c9c00f77-4f40-445c-b348-70754dfce19c
md"## Classification Model"

# ╔═╡ cd79f25b-36e0-463a-80b6-ebc622aa75d2
begin
	Core.eval(Main, :(import Flux, NNlib))
	classification_model = BSON.load("models/classification_model.bson")[:model]
	classification_model = classification_model |> gpu
	ŷ_classification = Neural.classify(classification_model, X) |> cpu
	classification_model
end

# ╔═╡ f8ec4620-fc02-4dca-b005-0961de8ed1af
Evaluation.rmse(y, ŷ_classification),
Evaluation.median_Δv(y, ŷ_classification),
Evaluation.mad_Δv(y, ŷ_classification),
Evaluation.cat_z_ratio(y, ŷ_classification)

# ╔═╡ e1ada0e9-c547-4642-92e6-71a8ef1ce5ad
begin
	density(y, label="Visual z", xlabel="z", ylabel="Density")
	density!(ŷ_classification, label="Classification z")
end

# ╔═╡ 8d64cf38-8af9-4b35-89b2-daa60fdab019
scatter(y, ŷ_classification, legend=:none, xlabel="Visual", ylabel="Classification")

# ╔═╡ Cell order:
# ╟─ed4d438e-6aaa-11eb-051e-efe644cce631
# ╠═6dc764b2-66e7-11eb-0833-9dc54a18f920
# ╠═21d5a8d1-d087-4598-851f-8cb8e67fee83
# ╠═8643602a-66e9-11eb-3700-374047551428
# ╟─646c8844-d25a-4453-a4d9-e6f6279c183b
# ╠═edd6e898-6797-11eb-2cee-791764fb425a
# ╠═7b0e34c6-67ba-11eb-2500-378603362df8
# ╠═e6d5d6fa-6aab-11eb-0434-19c6a5a97099
# ╠═5f25ed6f-ef8d-49c0-993d-e3d5c445fd99
# ╠═1d52576a-6abb-11eb-2dd8-c388b2365ddd
# ╟─c9c00f77-4f40-445c-b348-70754dfce19c
# ╠═cd79f25b-36e0-463a-80b6-ebc622aa75d2
# ╠═f8ec4620-fc02-4dca-b005-0961de8ed1af
# ╠═e1ada0e9-c547-4642-92e6-71a8ef1ce5ad
# ╠═8d64cf38-8af9-4b35-89b2-daa60fdab019
