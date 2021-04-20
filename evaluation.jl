### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ 364b618e-330c-4ae1-b70e-a8267028fee1
using BSON, FITSIO, Flux, Flux.Data, HDF5, Statistics, StatsPlots

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
end

# ╔═╡ 06f9cf4e-e341-410a-83a4-4d6a5ea5576a
begin
	density(y_validation, label="Visual", xlabel="z", ylabel="Density")
	density!(y_pipe, label="Pipeline")
end

# ╔═╡ 61d4b6de-5efa-400b-b4b0-cbccab2d9f6b
Evaluation.rmse(y_validation, y_pipe), Evaluation.catastrophic_redshift_ratio(y_validation, y_pipe)

# ╔═╡ 646c8844-d25a-4453-a4d9-e6f6279c183b
md"## Machine Learnig"

# ╔═╡ edd6e898-6797-11eb-2cee-791764fb425a
begin
	Core.eval(Main, :(import Flux, NNlib))
	model = gpu(BSON.load("models/model.bson")[:model])
	ŷ_train = cpu(Neural.predict(model, X_train))
	ŷ_validation = cpu(Neural.predict(model, X_validation))
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

# ╔═╡ 1d52576a-6abb-11eb-2dd8-c388b2365ddd
begin
	i = rand(1:size(id_validation, 2))
	z = y_validation[i]
	ẑ = ŷ_validation[i]
	Δv = Evaluation.compute_delta_v(z, ẑ)
	Utils.plot_spectrum(X_validation[:, i], label="z = $(z); ẑ = $(ẑ); Δv = $(Δv)")
	Utils.plot_spectral_lines!(z)
	Utils.plot_spectral_lines!(ẑ, color=:red, location=:bottom)
end

# ╔═╡ 6e379afc-6ac0-11eb-1c5a-7510123e34d2
md"## Approximate Inference in Bayesian Neural Network with Dropout

> Drawing a new function for each test point makes no difference if all we care about is obtaining the predictive mean and predictive variance
> (actually, for these two quantities this process is preferable to the one I will describe below),
> but this process does not result in draws from the induced distribution over functions.
> ([Uncertainty in Deep Learning](http://www.cs.ox.ac.uk/people/yarin.gal/website/blog_2248.html))"

# ╔═╡ 279ea616-6ac1-11eb-10b3-ff31e5c3305e
model_bayes = deepcopy(model)

# ╔═╡ 92607376-6ac1-11eb-3620-1ff033ef6890
function point_estimate(model, X; n_samples=32)
	trainmode!(model)
	outputs = reduce(hcat, [Neural.predict(model, X) for i in 1:n_samples])
	ŷ_mean = dropdims(mean(outputs, dims=2), dims=2)
	ŷ_std = dropdims(std(outputs, mean=ŷ_mean, dims=2), dims=2)
	return ŷ_mean, ŷ_std
end

# ╔═╡ 77682b6c-6ac2-11eb-2aa3-01e02dea681e
ŷ_bayes, σ_bayes = cpu(point_estimate(model_bayes, X_validation))

# ╔═╡ a8c710e6-6ac8-11eb-3f00-5d2b72864754
begin
	ŷ_error = y_validation - ŷ_bayes
	cor(abs.(ŷ_error), σ_bayes)
end

# ╔═╡ e396f046-71ce-11eb-1564-03d296917d94
density(σ_bayes, xlabel="σ", ylabel="Density", label=:none)

# ╔═╡ 92142f12-6acd-11eb-32c9-9f33783e75f4
σ_range = 0:0.001:maximum(σ_bayes)

# ╔═╡ c93ef038-6acc-11eb-2e6e-298f7178eb89
plot(
	σ_range,
	[Evaluation.rmse(y_validation[σ_bayes .< σ], ŷ_bayes[σ_bayes .< σ]) for σ in σ_range],
	ylabel="RMSE", xlabel="σ", label=:none)

# ╔═╡ fd62f99a-71cf-11eb-287c-475258d276aa
begin
	σ_threshold = 0.1
	idx = σ_bayes .< σ_threshold
	Evaluation.rmse(y_validation[idx], ŷ_bayes[idx]), Evaluation.catastrophic_redshift_ratio(y_validation[idx], ŷ_bayes[idx]), sum(idx) / length(y_validation)
end

# ╔═╡ bf58926c-6acd-11eb-3c23-bde13af4bfc2
begin
	vline([σ_threshold], label="Threshold")
	plot!(
		σ_range,
		[sum(σ_bayes .< σ) / length(σ_bayes) for σ in σ_range],
		label="Completeness", xlabel="σ")
	plot!(
		σ_range,
		[Evaluation.catastrophic_redshift_ratio(y_validation[σ_bayes .< σ], ŷ_bayes[σ_bayes .< σ])	for σ in σ_range],
		label="Catastrophic z Ratio", xlabel="σ")
end

# ╔═╡ Cell order:
# ╟─ed4d438e-6aaa-11eb-051e-efe644cce631
# ╠═364b618e-330c-4ae1-b70e-a8267028fee1
# ╠═6dc764b2-66e7-11eb-0833-9dc54a18f920
# ╠═8643602a-66e9-11eb-3700-374047551428
# ╠═06f9cf4e-e341-410a-83a4-4d6a5ea5576a
# ╠═61d4b6de-5efa-400b-b4b0-cbccab2d9f6b
# ╟─646c8844-d25a-4453-a4d9-e6f6279c183b
# ╠═edd6e898-6797-11eb-2cee-791764fb425a
# ╠═d554d1f7-93d5-445a-8a74-ef9035bb2190
# ╠═7b0e34c6-67ba-11eb-2500-378603362df8
# ╠═e6d5d6fa-6aab-11eb-0434-19c6a5a97099
# ╠═1d52576a-6abb-11eb-2dd8-c388b2365ddd
# ╟─6e379afc-6ac0-11eb-1c5a-7510123e34d2
# ╠═279ea616-6ac1-11eb-10b3-ff31e5c3305e
# ╠═92607376-6ac1-11eb-3620-1ff033ef6890
# ╠═77682b6c-6ac2-11eb-2aa3-01e02dea681e
# ╠═a8c710e6-6ac8-11eb-3f00-5d2b72864754
# ╠═e396f046-71ce-11eb-1564-03d296917d94
# ╠═92142f12-6acd-11eb-32c9-9f33783e75f4
# ╠═c93ef038-6acc-11eb-2e6e-298f7178eb89
# ╠═fd62f99a-71cf-11eb-287c-475258d276aa
# ╠═bf58926c-6acd-11eb-3c23-bde13af4bfc2
