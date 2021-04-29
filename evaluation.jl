### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# ╔═╡ 364b618e-330c-4ae1-b70e-a8267028fee1
using BSON, FITSIO, Flux, Flux.Data, HDF5, Printf, Statistics, StatsPlots

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
md"## Classical Model"

# ╔═╡ edd6e898-6797-11eb-2cee-791764fb425a
begin
	Core.eval(Main, :(import Flux, NNlib))
	model = gpu(BSON.load("models/model.bson")[:model])
	ŷ_train = cpu(Neural.predict(model, X_train))
	ŷ_validation = cpu(Neural.predict(model, X_validation))
	model
end

# ╔═╡ d554d1f7-93d5-445a-8a74-ef9035bb2190
Evaluation.rmse(y_train, ŷ_train), Evaluation.catastrophic_redshift_ratio(y_train, ŷ_train)

# ╔═╡ 7b0e34c6-67ba-11eb-2500-378603362df8
Evaluation.rmse(y_validation, ŷ_validation), Evaluation.catastrophic_redshift_ratio(y_validation, ŷ_validation)

# ╔═╡ e6d5d6fa-6aab-11eb-0434-19c6a5a97099
begin
	density(y_validation, label="Visual", xlabel="z", ylabel="Density")
	density!(ŷ_validation, label="Model")
end

# ╔═╡ 5f25ed6f-ef8d-49c0-993d-e3d5c445fd99
begin
	scatter_model = scatter(y_validation, ŷ_validation, legend=:none)
	scatter_pipe = scatter(y_validation, y_pipe, legend=:none)
	scatters = @layout [a b]
	plot(scatter_model, scatter_pipe, layout=scatters)
end

# ╔═╡ c05e8b4a-32d7-49b2-8a38-b7f040e1921c
begin
	threshold_range = 1000:10:5000
	plot(
		threshold_range,
		[Evaluation.catastrophic_redshift_ratio(y_validation, ŷ_validation, threshold=t) for t in threshold_range],
		label="Model", xlabel="Δv", ylabel="Cat. z")
	plot!(
		threshold_range,
		[Evaluation.catastrophic_redshift_ratio(y_validation, y_pipe, threshold=t) for t in threshold_range],
		label="Pipeline")
end

# ╔═╡ 1d52576a-6abb-11eb-2dd8-c388b2365ddd
begin	
	Δv = Evaluation.compute_delta_v(y_validation, ŷ_validation)
	# random
	i = rand(1:size(id_validation, 2))
	# cat. z
	#i = rand((1:size(id_validation, 2))[Δv .> 3000])
	# absolute error
	#i = sortperm(abs.(y_validation - ŷ_validation))[end]

	z = y_validation[i]
	ẑ = ŷ_validation[i]
	title = @sprintf "z = %.3f; ẑ = %.3f; Δv = %.3f" z ẑ Δv[i]
	Utils.plot_spectrum(X_validation[:, i], title=title, legend=:none)
	Utils.plot_spectral_lines!(z)
	prepared_spectrum = Utils.plot_spectral_lines!(ẑ, color=:red, location=:bottom)
	loglam, flux = Utils.get_spectrum("Superset_DR12Q", id_validation[:, i]...)
	original_spectrum = plot(10 .^ loglam, flux, legend=:none)
	l = @layout [a; b]
	plot(prepared_spectrum, original_spectrum, layout=l)
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
	outputs = reduce(hcat, [Neural.predict(model, X) for i in 1:n_samples])
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
# ╟─fe9a448e-fc02-4d70-b12b-954cdf472849
# ╠═92607376-6ac1-11eb-3620-1ff033ef6890
# ╠═08c367cc-3c81-450c-97c5-96c6b80c7a0f
# ╠═a8c710e6-6ac8-11eb-3f00-5d2b72864754
# ╠═e396f046-71ce-11eb-1564-03d296917d94
# ╠═92142f12-6acd-11eb-32c9-9f33783e75f4
# ╠═c93ef038-6acc-11eb-2e6e-298f7178eb89
# ╠═bf58926c-6acd-11eb-3c23-bde13af4bfc2
