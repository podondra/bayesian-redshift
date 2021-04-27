### A Pluto.jl notebook ###
# v0.14.3

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
end

# ╔═╡ 06f9cf4e-e341-410a-83a4-4d6a5ea5576a
begin
	density(y_validation, label="Visual", xlabel="z", ylabel="Density")
	density!(y_pipe, label="Pipeline")
end

# ╔═╡ 61d4b6de-5efa-400b-b4b0-cbccab2d9f6b
Evaluation.rmse(y_validation, y_pipe), Evaluation.catastrophic_redshift_ratio(y_validation, y_pipe)

# ╔═╡ 646c8844-d25a-4453-a4d9-e6f6279c183b
md"## Machine Learning

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

# ╔═╡ edd6e898-6797-11eb-2cee-791764fb425a
begin
	Core.eval(Main, :(import Flux, NNlib))
	model = gpu(BSON.load("models/model.bson")[:model])
	ŷ_train = cpu(Neural.predict(model, X_train))
	ŷ_train, σ_train = cpu(point_estimate(model, X_train))
	ŷ_validation, σ_validation = cpu(point_estimate(model, X_validation))
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

# ╔═╡ 450a8f33-7110-435f-b9b3-7a0699a9ada0
begin
	# random
	#i = rand(1:size(id_validation, 2))
	# cat. z
	Δv = Evaluation.compute_delta_v(y_validation, ŷ_validation)
	i = rand((1:size(id_validation, 2))[Δv .> 3000])
	# absolute error
	#i = sortperm(abs.(y_validation - ŷ_validation))[end]
end

# ╔═╡ 1d52576a-6abb-11eb-2dd8-c388b2365ddd
begin
	z = y_validation[i]
	ẑ = ŷ_validation[i]
	σ = σ_validation[i]
	title = @sprintf "z = %.3f; ẑ = %.3f; Δv = %.3f; σ = %.3f" z ẑ Δv[i] σ
	Utils.plot_spectrum(X_validation[:, i], title=title, legend=:none)
	Utils.plot_spectral_lines!(z)
	prepared_spectrum = Utils.plot_spectral_lines!(ẑ, color=:red, location=:bottom)
	loglam, flux = Utils.get_spectrum("Superset_DR12Q", id_validation[:, i]...)
	original_spectrum = plot(10 .^ loglam, flux, legend=:none)
	l = @layout [a; b]
	plot(prepared_spectrum, original_spectrum, layout=l)
end

# ╔═╡ a8c710e6-6ac8-11eb-3f00-5d2b72864754
begin
	ŷ_error = y_validation - ŷ_validation
	cor(abs.(ŷ_error), σ_validation)
end

# ╔═╡ e396f046-71ce-11eb-1564-03d296917d94
histogram(σ_validation, xlabel="σ", ylabel="Density", label=:none)

# ╔═╡ 92142f12-6acd-11eb-32c9-9f33783e75f4
σ_range = 0:0.001:maximum(σ_validation)

# ╔═╡ c93ef038-6acc-11eb-2e6e-298f7178eb89
plot(
	σ_range,
	[Evaluation.rmse(y_validation[σ_validation .< σ],
	 ŷ_validation[σ_validation .< σ]) for σ in σ_range],
	ylabel="RMSE", xlabel="σ", label=:none)

# ╔═╡ fd62f99a-71cf-11eb-287c-475258d276aa
begin
	σ_threshold = 0.1
	idx = σ_validation .< σ_threshold
	Evaluation.rmse(y_validation[idx], ŷ_validation[idx]),
	Evaluation.catastrophic_redshift_ratio(y_validation[idx],
	ŷ_validation[idx]), sum(idx) / length(y_validation)
end

# ╔═╡ bf58926c-6acd-11eb-3c23-bde13af4bfc2
begin
	plot(
		σ_range,
		[sum(σ_validation .< σ) / length(σ_validation) for σ in σ_range],
		label="Completeness", xlabel="σ")
	plot!(
		σ_range,
		[Evaluation.catastrophic_redshift_ratio(
				y_validation[σ_validation .< σ], ŷ_validation[σ_validation .< σ])
			for σ in σ_range],
		label="Catastrophic z Ratio", xlabel="σ")
	vline!([σ_threshold], label="Threshold")
end

# ╔═╡ Cell order:
# ╟─ed4d438e-6aaa-11eb-051e-efe644cce631
# ╠═364b618e-330c-4ae1-b70e-a8267028fee1
# ╠═6dc764b2-66e7-11eb-0833-9dc54a18f920
# ╠═8643602a-66e9-11eb-3700-374047551428
# ╠═06f9cf4e-e341-410a-83a4-4d6a5ea5576a
# ╠═61d4b6de-5efa-400b-b4b0-cbccab2d9f6b
# ╟─646c8844-d25a-4453-a4d9-e6f6279c183b
# ╠═92607376-6ac1-11eb-3620-1ff033ef6890
# ╠═edd6e898-6797-11eb-2cee-791764fb425a
# ╠═d554d1f7-93d5-445a-8a74-ef9035bb2190
# ╠═7b0e34c6-67ba-11eb-2500-378603362df8
# ╠═e6d5d6fa-6aab-11eb-0434-19c6a5a97099
# ╠═450a8f33-7110-435f-b9b3-7a0699a9ada0
# ╠═1d52576a-6abb-11eb-2dd8-c388b2365ddd
# ╠═a8c710e6-6ac8-11eb-3f00-5d2b72864754
# ╠═e396f046-71ce-11eb-1564-03d296917d94
# ╠═92142f12-6acd-11eb-32c9-9f33783e75f4
# ╠═c93ef038-6acc-11eb-2e6e-298f7178eb89
# ╠═fd62f99a-71cf-11eb-287c-475258d276aa
# ╠═bf58926c-6acd-11eb-3c23-bde13af4bfc2
