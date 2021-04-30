### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# ╔═╡ 364b618e-330c-4ae1-b70e-a8267028fee1
using BSON, CUDA, FITSIO, Flux, Flux.Data, HDF5, Printf, Statistics, StatsPlots

# ╔═╡ 6dc764b2-66e7-11eb-0833-9dc54a18f920
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
	id_validation = read(dr12q_file, "id_va")
	X_train = gpu(read(dr12q_file, "X_tr"))
	X_validation = gpu(read(dr12q_file, "X_va"))
	y_train = read(dr12q_file, "z_vi_tr")
	y_validation = read(dr12q_file, "z_vi_va")
	y_pipe = read(dr12q_file, "z_pipe_va")
	close(dr12q_file)
	size(X_train), size(X_validation)
end

# ╔═╡ 9bc66b89-d566-4335-a257-e228ff9d7e3e
CUDA.versioninfo()

# ╔═╡ 59c68c29-51c2-4cdd-9e74-567bc5a2cded
md"## Pipeline Baseline"

# ╔═╡ 06f9cf4e-e341-410a-83a4-4d6a5ea5576a
begin
	density(y_validation, label="Visual", xlabel="z", ylabel="Density")
	density!(y_pipe, label="Pipeline")
end

# ╔═╡ 61d4b6de-5efa-400b-b4b0-cbccab2d9f6b
Evaluation.rmse(y_validation, y_pipe), Evaluation.catastrophic_redshift_ratio(y_validation, y_pipe)

# ╔═╡ 646c8844-d25a-4453-a4d9-e6f6279c183b
md"## Regression Model"

# ╔═╡ edd6e898-6797-11eb-2cee-791764fb425a
begin
	Core.eval(Main, :(import Flux, NNlib))
	model_reg = gpu(BSON.load("models/regression_model.bson")[:model])
	ŷ_train_reg = cpu(Neural.regress(model_reg, X_train))
	ŷ_validation_reg = cpu(Neural.regress(model_reg, X_validation))
	model_reg
end

# ╔═╡ d554d1f7-93d5-445a-8a74-ef9035bb2190
Evaluation.rmse(y_train, ŷ_train_reg),
Evaluation.catastrophic_redshift_ratio(y_train, ŷ_train_reg)

# ╔═╡ 7b0e34c6-67ba-11eb-2500-378603362df8
Evaluation.rmse(y_validation, ŷ_validation_reg), Evaluation.catastrophic_redshift_ratio(y_validation, ŷ_validation_reg)

# ╔═╡ e6d5d6fa-6aab-11eb-0434-19c6a5a97099
begin
	density(y_validation, label="Visual", xlabel="z", ylabel="Density")
	density!(ŷ_validation_reg, label="Model")
end

# ╔═╡ 5f25ed6f-ef8d-49c0-993d-e3d5c445fd99
begin
	scatter_model_reg = scatter(y_validation, ŷ_validation_reg, legend=:none)
	scatter_pipe = scatter(y_validation, y_pipe, legend=:none)
	plot(scatter_model_reg, scatter_pipe, layout=@layout [a b])
end

# ╔═╡ c05e8b4a-32d7-49b2-8a38-b7f040e1921c
begin
	threshold_range = 1000:10:5000
	plot(
		threshold_range,
		[Evaluation.catastrophic_redshift_ratio(
				y_validation, ŷ_validation_reg, threshold=t)
			for t in threshold_range],
		label="Model", xlabel="Δv", ylabel="Cat. z")
	plot!(
		threshold_range,
		[Evaluation.catastrophic_redshift_ratio(y_validation, y_pipe, threshold=t) for t in threshold_range],
		label="Pipeline")
end

# ╔═╡ 1d52576a-6abb-11eb-2dd8-c388b2365ddd
begin	
	Δv_reg = Evaluation.compute_delta_v(y_validation, ŷ_validation_reg)
	# random
	i = rand(1:size(id_validation, 2))
	# cat. z
	#i = rand((1:size(id_validation, 2))[Δv .> 3000])
	# absolute error
	#i = sortperm(abs.(y_validation - ŷ_validation))[end]

	z = y_validation[i]
	ẑ = ŷ_validation_reg[i]
	title = @sprintf "z = %.3f; ẑ = %.3f; Δv = %.3f" z ẑ Δv_reg[i]
	Utils.plot_spectrum(X_validation[:, i], title=title, legend=:none)
	Utils.plot_spectral_lines!(z)
	prepared_spectrum = Utils.plot_spectral_lines!(ẑ, color=:red, location=:bottom)
	loglam, flux = Utils.get_spectrum("Superset_DR12Q", id_validation[:, i]...)
	original_spectrum = plot(10 .^ loglam, flux, legend=:none)
	l = @layout [a; b]
	plot(prepared_spectrum, original_spectrum, layout=l)
end

# ╔═╡ c9c00f77-4f40-445c-b348-70754dfce19c
md"## Classification Model"

# ╔═╡ cd79f25b-36e0-463a-80b6-ebc622aa75d2
begin
	Core.eval(Main, :(import Flux, NNlib))
	model_clf = gpu(BSON.load("models/classification_model.bson")[:model])
	ŷ_train_clf = cpu(Neural.classify(model_clf, X_train))
	ŷ_validation_clf = cpu(Neural.classify(model_clf, X_validation))
	model_clf
end

# ╔═╡ e1d4a0f3-e1cb-4576-ae6a-97e735438236
Evaluation.rmse(y_train, ŷ_train_clf),
Evaluation.catastrophic_redshift_ratio(y_train, ŷ_train_clf)

# ╔═╡ f8ec4620-fc02-4dca-b005-0961de8ed1af
Evaluation.rmse(y_validation, ŷ_validation_clf), Evaluation.catastrophic_redshift_ratio(y_validation, ŷ_validation_clf)

# ╔═╡ e1ada0e9-c547-4642-92e6-71a8ef1ce5ad
begin
	density(y_validation, label="Visual", xlabel="z", ylabel="Density")
	density!(ŷ_validation_clf, label="Model")
end

# ╔═╡ 8d64cf38-8af9-4b35-89b2-daa60fdab019
begin
	scatter_model_clf = scatter(y_validation, ŷ_validation_clf, legend=:none)
	plot(scatter_model_reg, scatter_pipe, layout=@layout [a b])
end

# ╔═╡ fe9a448e-fc02-4d70-b12b-954cdf472849
md"## Bayesian Model

> Drawing a new function for each test point makes no difference if all we care about is obtaining the predictive mean and predictive variance
> (actually, for these two quantities this process is preferable to the one I will describe below),
> but this process does not result in draws from the induced distribution over functions.
> ([Uncertainty in Deep Learning](http://www.cs.ox.ac.uk/people/yarin.gal/website/blog_2248.html))"

# ╔═╡ 92607376-6ac1-11eb-3620-1ff033ef6890
function point_estimate(model, X; n_samples=64)
	trainmode!(model)
	outputs = reduce(hcat, [Neural.regress(model, X) for i in 1:n_samples])
	ŷ_mean = dropdims(mean(outputs, dims=2), dims=2)
	ŷ_std = dropdims(std(outputs, mean=ŷ_mean, dims=2), dims=2)
	return ŷ_mean, ŷ_std
end

# ╔═╡ 08c367cc-3c81-450c-97c5-96c6b80c7a0f
begin
	Core.eval(Main, :(import Flux, NNlib))
	model_bayes = gpu(BSON.load("models/bayesian_model.bson")[:model])
	ŷ_train_bayes, σ_train = cpu(point_estimate(model_bayes, X_train))
	ŷ_validation_bayes, σ_validation = cpu(point_estimate(model_bayes, X_validation))
	model_bayes
end

# ╔═╡ 895d5f6b-88e6-4fa9-a9d9-0516e27fe30c
Evaluation.rmse(y_train, ŷ_train_bayes),
Evaluation.catastrophic_redshift_ratio(y_train, ŷ_train_bayes)

# ╔═╡ 4d920462-4f1f-47a6-b0b1-dd3a656251c9
Evaluation.rmse(y_validation, ŷ_validation_bayes), Evaluation.catastrophic_redshift_ratio(y_validation, ŷ_validation_bayes)

# ╔═╡ a8c710e6-6ac8-11eb-3f00-5d2b72864754
begin
	ŷ_error_bayes = y_validation - ŷ_validation_bayes
	cor(abs.(ŷ_error_bayes), σ_validation)
end

# ╔═╡ e396f046-71ce-11eb-1564-03d296917d94
histogram(σ_validation, xlabel="σ", ylabel="Density", label=:none)

# ╔═╡ 92142f12-6acd-11eb-32c9-9f33783e75f4
σ_range = 0:0.001:maximum(σ_validation)

# ╔═╡ c93ef038-6acc-11eb-2e6e-298f7178eb89
plot(
	σ_range,
	[Evaluation.rmse(y_validation[σ_validation .< σ],
	 ŷ_validation_bayes[σ_validation .< σ]) for σ in σ_range],
	ylabel="RMSE", xlabel="σ", label=:none)

# ╔═╡ bf58926c-6acd-11eb-3c23-bde13af4bfc2
begin
	plot(
		σ_range,
		[sum(σ_validation .< σ) / length(σ_validation) for σ in σ_range],
		label="Completeness", xlabel="σ")
	plot!(
		σ_range,
		[Evaluation.catastrophic_redshift_ratio(
				y_validation[σ_validation .< σ],
				ŷ_validation_bayes[σ_validation .< σ])
			for σ in σ_range],
		label="Catastrophic z Ratio", xlabel="σ")
end

# ╔═╡ Cell order:
# ╟─ed4d438e-6aaa-11eb-051e-efe644cce631
# ╠═364b618e-330c-4ae1-b70e-a8267028fee1
# ╠═6dc764b2-66e7-11eb-0833-9dc54a18f920
# ╠═8643602a-66e9-11eb-3700-374047551428
# ╠═9bc66b89-d566-4335-a257-e228ff9d7e3e
# ╟─59c68c29-51c2-4cdd-9e74-567bc5a2cded
# ╠═06f9cf4e-e341-410a-83a4-4d6a5ea5576a
# ╠═61d4b6de-5efa-400b-b4b0-cbccab2d9f6b
# ╟─646c8844-d25a-4453-a4d9-e6f6279c183b
# ╠═edd6e898-6797-11eb-2cee-791764fb425a
# ╠═d554d1f7-93d5-445a-8a74-ef9035bb2190
# ╠═7b0e34c6-67ba-11eb-2500-378603362df8
# ╠═e6d5d6fa-6aab-11eb-0434-19c6a5a97099
# ╠═5f25ed6f-ef8d-49c0-993d-e3d5c445fd99
# ╠═c05e8b4a-32d7-49b2-8a38-b7f040e1921c
# ╠═1d52576a-6abb-11eb-2dd8-c388b2365ddd
# ╟─c9c00f77-4f40-445c-b348-70754dfce19c
# ╠═cd79f25b-36e0-463a-80b6-ebc622aa75d2
# ╠═e1d4a0f3-e1cb-4576-ae6a-97e735438236
# ╠═f8ec4620-fc02-4dca-b005-0961de8ed1af
# ╠═e1ada0e9-c547-4642-92e6-71a8ef1ce5ad
# ╠═8d64cf38-8af9-4b35-89b2-daa60fdab019
# ╟─fe9a448e-fc02-4d70-b12b-954cdf472849
# ╠═92607376-6ac1-11eb-3620-1ff033ef6890
# ╠═08c367cc-3c81-450c-97c5-96c6b80c7a0f
# ╠═895d5f6b-88e6-4fa9-a9d9-0516e27fe30c
# ╠═4d920462-4f1f-47a6-b0b1-dd3a656251c9
# ╠═a8c710e6-6ac8-11eb-3f00-5d2b72864754
# ╠═e396f046-71ce-11eb-1564-03d296917d94
# ╠═92142f12-6acd-11eb-32c9-9f33783e75f4
# ╠═c93ef038-6acc-11eb-2e6e-298f7178eb89
# ╠═bf58926c-6acd-11eb-3c23-bde13af4bfc2
