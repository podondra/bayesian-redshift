### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# ╔═╡ 6dc764b2-66e7-11eb-0833-9dc54a18f920
begin
	using BSON, CUDA, FITSIO, Flux, Flux.Data, HDF5, Printf, Statistics, StatsBase, StatsPlots
	include("Evaluation.jl"); using .Evaluation
	include("Neural.jl"); using .Neural
	include("Utils.jl"); using .Utils
	CUDA.versioninfo()
end

# ╔═╡ ed4d438e-6aaa-11eb-051e-efe644cce631
md"# Evaluation"

# ╔═╡ 8643602a-66e9-11eb-3700-374047551428
begin
	dr12q_file = h5open("data/dr12q_superset.hdf5", "r")
	id_va = read(dr12q_file, "id_va")
	X_tr = read(dr12q_file, "X_tr") |> gpu
	X_va = read(dr12q_file, "X_va") |> gpu
	y_tr = read(dr12q_file, "z_vi_tr")
	y_va = read(dr12q_file, "z_vi_va")
	y_pipe = read(dr12q_file, "z_pipe_va")
	close(dr12q_file)
	size(X_tr), size(X_va)
end

# ╔═╡ 59c68c29-51c2-4cdd-9e74-567bc5a2cded
md"## Pipeline Baseline"

# ╔═╡ 06f9cf4e-e341-410a-83a4-4d6a5ea5576a
begin
	density(y_va, label="Visual", xlabel="z", ylabel="Density")
	density!(y_pipe, label="Pipeline")
end

# ╔═╡ 61d4b6de-5efa-400b-b4b0-cbccab2d9f6b
Evaluation.rmse(y_va, y_pipe), Evaluation.cat_z_ratio(y_va, y_pipe)

# ╔═╡ 19272ada-0111-4000-9c6a-f0f91a1973ff
scatter(y_va, y_pipe, legend=:none, xlabel="Visual", ylabel="Pipeline")

# ╔═╡ 646c8844-d25a-4453-a4d9-e6f6279c183b
md"## Regression Model"

# ╔═╡ edd6e898-6797-11eb-2cee-791764fb425a
begin
	Core.eval(Main, :(import Flux, NNlib))
	model_reg = BSON.load("models/regression_model.bson")[:model] |> gpu
	ŷ_tr_reg = Neural.regress(model_reg, X_tr) |> cpu
	ŷ_va_reg = Neural.regress(model_reg, X_va) |> cpu
	model_reg
end

# ╔═╡ d554d1f7-93d5-445a-8a74-ef9035bb2190
Evaluation.rmse(y_tr, ŷ_tr_reg), Evaluation.cat_z_ratio(y_tr, ŷ_tr_reg)

# ╔═╡ 7b0e34c6-67ba-11eb-2500-378603362df8
Evaluation.rmse(y_va, ŷ_va_reg), Evaluation.cat_z_ratio(y_va, ŷ_va_reg)

# ╔═╡ e6d5d6fa-6aab-11eb-0434-19c6a5a97099
begin
	density(y_va, label="Visual", xlabel="z", ylabel="Density")
	density!(ŷ_va_reg, label="Regression")
end

# ╔═╡ 5f25ed6f-ef8d-49c0-993d-e3d5c445fd99
scatter(y_va, ŷ_va_reg, legend=:none, xlabel="Visual", ylabel="Regression")

# ╔═╡ c05e8b4a-32d7-49b2-8a38-b7f040e1921c
begin
	thresholds = 1000:10:5000
	plot(
		thresholds,
		[Evaluation.cat_z_ratio(y_va, ŷ_va_reg, threshold=t) for t in thresholds],
		label="Regression", xlabel="Δv", ylabel="Cat. z")
	plot!(
		thresholds,
		[Evaluation.cat_z_ratio(y_va, y_pipe, threshold=t) for t in thresholds],
		label="Pipeline")
end

# ╔═╡ 1d52576a-6abb-11eb-2dd8-c388b2365ddd
begin	
	Δv_reg = Evaluation.compute_delta_v(y_va, ŷ_va_reg)
	# random
	i = rand(1:size(id_va, 2))
	# cat. z
	i = rand((1:size(id_va, 2))[Δv_reg .> 3000])
	# absolute error
	i = sortperm(abs.(y_va - ŷ_va_reg))[end]

	title_reg = @sprintf(
		"z = %.3f; ẑ = %.3f; Δv = %.3f", y_va[i], ŷ_va_reg[i], Δv_reg[i])
	Utils.plot_spectrum(X_va[:, i], title=title_reg, legend=:none)
	Utils.plot_spectral_lines!(y_va[i])
	prep_spec_reg = Utils.plot_spectral_lines!(
		ŷ_va_reg[i], color=:red, location=:bottom)
	orig_spec_reg = plot(
		Utils.get_spectrum("Superset_DR12Q", id_va[:, i]...)..., legend=:none)
	plot(prep_spec_reg, orig_spec_reg, layout=@layout [a; b])
end

# ╔═╡ c9c00f77-4f40-445c-b348-70754dfce19c
md"## Classification Model"

# ╔═╡ cd79f25b-36e0-463a-80b6-ebc622aa75d2
begin
	Core.eval(Main, :(import Flux, NNlib))
	model_clf = BSON.load("models/classification_model.bson")[:model] |> gpu
	ŷ_tr_clf = Neural.classify(model_clf, X_tr) |> cpu
	ŷ_va_clf = Neural.classify(model_clf, X_va) |> cpu
	model_clf
end

# ╔═╡ e1d4a0f3-e1cb-4576-ae6a-97e735438236
Evaluation.rmse(y_tr, ŷ_tr_clf), Evaluation.cat_z_ratio(y_tr, ŷ_tr_clf)

# ╔═╡ f8ec4620-fc02-4dca-b005-0961de8ed1af
Evaluation.rmse(y_va, ŷ_va_clf), Evaluation.cat_z_ratio(y_va, ŷ_va_clf)

# ╔═╡ e1ada0e9-c547-4642-92e6-71a8ef1ce5ad
begin
	density(y_va, label="Visual", xlabel="z", ylabel="Density")
	density!(ŷ_va_clf, label="Classification")
end

# ╔═╡ 8d64cf38-8af9-4b35-89b2-daa60fdab019
scatter(y_va, ŷ_va_clf, legend=:none, xlabel="Visual", ylabel="Classification")

# ╔═╡ 6cee541a-89a7-40fe-8567-ed6c9fa80fea
begin
	Δv_clf = Evaluation.compute_delta_v(y_va, ŷ_va_clf)
	# random
	j = rand(1:size(id_va, 2))
	# cat. z
	j = rand((1:size(id_va, 2))[Δv_clf .> 3000])
	# absolute error
	#j = sortperm(abs.(y_va - ŷ_va_clf))[end]

	title_clf = @sprintf(
		"z = %.3f; ẑ = %.3f; Δv = %.3f", y_va[j], ŷ_va_clf[j], Δv_clf[j])
	Utils.plot_spectrum(X_va[:, j], title=title_clf, legend=:none)
	Utils.plot_spectral_lines!(y_va[j])
	prep_spec_clf = Utils.plot_spectral_lines!(
		ŷ_va_clf[j], color=:red, location=:bottom)
	loglam_clf, flux_clf = Utils.get_spectrum("Superset_DR12Q", id_va[:, j]...)
	orig_spec_clf = plot(10 .^ loglam_clf, flux_clf, legend=:none)
	plot(prep_spec_clf, orig_spec_clf, layout=@layout [a; b])
end

# ╔═╡ fe9a448e-fc02-4d70-b12b-954cdf472849
md"## Bayesian Regression Model

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
	model_bayes = BSON.load("models/bayesian_model.bson")[:model] |> gpu
	ŷ_tr_bayes, σ_tr = point_estimate(model_bayes, X_tr) |> cpu
	ŷ_va_bayes, σ_va = point_estimate(model_bayes, X_va) |> cpu
	model_bayes
end

# ╔═╡ 895d5f6b-88e6-4fa9-a9d9-0516e27fe30c
Evaluation.rmse(y_tr, ŷ_tr_bayes), Evaluation.cat_z_ratio(y_tr, ŷ_tr_bayes)

# ╔═╡ 4d920462-4f1f-47a6-b0b1-dd3a656251c9
Evaluation.rmse(y_va, ŷ_va_bayes), Evaluation.cat_z_ratio(y_va, ŷ_va_bayes)

# ╔═╡ a8c710e6-6ac8-11eb-3f00-5d2b72864754
begin
	ŷ_error_bayes = y_va - ŷ_va_bayes
	cor(abs.(ŷ_error_bayes), σ_va)
end

# ╔═╡ e396f046-71ce-11eb-1564-03d296917d94
histogram(σ_va, xlabel="σ", ylabel="Count", label=:none)

# ╔═╡ 92142f12-6acd-11eb-32c9-9f33783e75f4
σs = 0:0.001:maximum(σ_va)

# ╔═╡ c93ef038-6acc-11eb-2e6e-298f7178eb89
plot(
	σs,	[Evaluation.rmse(y_va[σ_va .< σ], ŷ_va_bayes[σ_va .< σ]) for σ in σs],
	ylabel="RMSE", xlabel="σ", label=:none)

# ╔═╡ bf58926c-6acd-11eb-3c23-bde13af4bfc2
begin
	plot(
		σs, [sum(σ_va .< σ) / length(σ_va) for σ in σs],
		label="Completeness", xlabel="σ")
	plot!(
		σs,
		[Evaluation.cat_z_ratio(y_va[σ_va .< σ], ŷ_va_bayes[σ_va .< σ]) for σ in σs],
		label="Catastrophic z Ratio", xlabel="σ")
end

# ╔═╡ 02c2ae29-6949-4de4-80a7-59223bd2233c
md"## Bayesian Classification Model"

# ╔═╡ d0fe4d01-006a-451e-ad4d-558b0136a368
md"### Variation Ratio"

# ╔═╡ daee4dfd-83d1-40be-a8b1-35bfed361c3c
begin
	trainmode!(model_clf)
	batch_size = 256
	ŷ_va = zeros(size(X_va, 2))
	variation_ratios = zeros(size(X_va, 2))
	for idx in 1:size(X_va, 2)
		output = model_clf(X_va[:, [idx for i in 1:batch_size]]) |> cpu
		ŷ_va[idx] = findmax(countmap(Flux.onecold(output, 0.0f0:0.01f0:5.98f0)))[2]
		variation_ratios[idx] = 1 - maximum(proportions(Flux.onecold(output)))
	end
	Evaluation.rmse(y_va, ŷ_va), Evaluation.cat_z_ratio(y_va, ŷ_va)
end

# ╔═╡ 04bfd3c5-f876-4b69-b992-6719f07fc86d
histogram(variation_ratios, xlabel="Variation Ratio", ylabel="Count", label=:none)

# ╔═╡ 92e96e12-a833-48c1-b700-b2d36f5f9fa4
ts = 0:0.001:1

# ╔═╡ 2cbbbc35-fef9-4475-b7e9-8356d4d30c14
plot(ts, [Evaluation.rmse(y_va[variation_ratios .< t], ŷ_va[variation_ratios .< t]) for t in ts], ylabel="RMSE", xlabel="Threshold", label=:none)

# ╔═╡ 98759103-0f1f-4e39-9c1f-f162979645cf
begin
	plot(
		plot(
			ts, [sum(variation_ratios .< t) / length(variation_ratios) for t in ts],
			label="Completeness", xlabel="Threshold"),
		plot(
			ts, [Evaluation.cat_z_ratio(y_va[variation_ratios .< t], ŷ_va[variation_ratios .< t]) for t in ts],
			label="Catastrophic z Ratio", xlabel="Threshold"),
		layout=@layout [a; b])
end

# ╔═╡ Cell order:
# ╟─ed4d438e-6aaa-11eb-051e-efe644cce631
# ╠═6dc764b2-66e7-11eb-0833-9dc54a18f920
# ╠═8643602a-66e9-11eb-3700-374047551428
# ╟─59c68c29-51c2-4cdd-9e74-567bc5a2cded
# ╠═06f9cf4e-e341-410a-83a4-4d6a5ea5576a
# ╠═61d4b6de-5efa-400b-b4b0-cbccab2d9f6b
# ╠═19272ada-0111-4000-9c6a-f0f91a1973ff
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
# ╠═6cee541a-89a7-40fe-8567-ed6c9fa80fea
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
# ╟─02c2ae29-6949-4de4-80a7-59223bd2233c
# ╟─d0fe4d01-006a-451e-ad4d-558b0136a368
# ╠═daee4dfd-83d1-40be-a8b1-35bfed361c3c
# ╠═04bfd3c5-f876-4b69-b992-6719f07fc86d
# ╠═92e96e12-a833-48c1-b700-b2d36f5f9fa4
# ╠═2cbbbc35-fef9-4475-b7e9-8356d4d30c14
# ╠═98759103-0f1f-4e39-9c1f-f162979645cf
