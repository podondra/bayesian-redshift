### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 6dc764b2-66e7-11eb-0833-9dc54a18f920
begin
	using BSON
	using FITSIO
	using Flux
	using Flux.Data
	using HDF5
	using Statistics
	using StatsPlots
	include("Evaluation.jl"); import .Evaluation
	include("Neural.jl"); import .Neural
	include("Utils.jl"); import .Utils
end

# ╔═╡ ed4d438e-6aaa-11eb-051e-efe644cce631
md"# Evaluation"

# ╔═╡ 8643602a-66e9-11eb-3700-374047551428
begin
	datafile = h5open("data/dr16q_superset.hdf5")
	id = read(datafile, "id_va")
	X = read(datafile, "X_va") |> gpu
	y = convert(Array{Float32}, read(datafile, "z_va"))
	close(datafile)
	size(id), typeof(id), size(X), typeof(X), size(y), typeof(y)
end

# ╔═╡ edd6e898-6797-11eb-2cee-791764fb425a
begin
	Core.eval(Main, :(import Flux, NNlib))
	nn = BSON.load("models/nn.bson")[:model] |> gpu
	vgg11 = BSON.load("models/vgg11.bson")[:model] |> gpu
	nn, vgg11
end

# ╔═╡ f614d7c0-6aaa-11eb-1c62-73c54a75a0ab
md"## Fully-Connected Neural Network"

# ╔═╡ 9e6ee084-6798-11eb-3683-b5c8c998cd77
ŷ_nn = Neural.predict(nn, X) |> cpu

# ╔═╡ 06de9ac0-679d-11eb-17d8-6f5359328d71
Evaluation.rmse(y, ŷ_nn)

# ╔═╡ 67e19e7e-679f-11eb-0eb1-c39fc5eac3ed
Evaluation.catastrophic_redshift_ratio(y, ŷ_nn)

# ╔═╡ 33c2836a-6aab-11eb-1c93-257ed54b0585
begin
	density(y, label="Validation Set", xlabel="z", ylabel="Density")
	density!(ŷ_nn, label="Fully-Connected Predictions")
end

# ╔═╡ 0348b7e0-6aab-11eb-16ba-d574e6296961
md"## Convolutional Neural Network"

# ╔═╡ 7b0e34c6-67ba-11eb-2500-378603362df8
ŷ_vgg11 = Neural.predict(vgg11, X) |> cpu

# ╔═╡ 89f370c6-67ba-11eb-24da-2b0a0f0ad532
Evaluation.rmse(y, ŷ_vgg11)

# ╔═╡ 98d6ed82-6ac3-11eb-0851-89add7ad3b38
Evaluation.catastrophic_redshift_ratio(y, ŷ_vgg11)

# ╔═╡ 8d836ffc-67ba-11eb-0d82-f70af20aeeae
Evaluation.catastrophic_redshift_ratio(y, ŷ_vgg11, threshold=6000)

# ╔═╡ 1d52576a-6abb-11eb-2dd8-c388b2365ddd
begin
	rnd_i = rand(1:size(id, 1))
	loglam, flux = Utils.get_spectrum(id[:, rnd_i]...)
	plot(10 .^ loglam, flux)
	Utils.plot_spectral_lines!(y[rnd_i])
	Utils.plot_spectral_lines!(ŷ_vgg11[rnd_i], color=:red, location=:bottom)
end

# ╔═╡ e6d5d6fa-6aab-11eb-0434-19c6a5a97099
begin
	density(y, label="Validation Set", xlabel="z", ylabel="Density")
	density!(ŷ_vgg11, label="VGG11 Predictions")
end

# ╔═╡ 6e379afc-6ac0-11eb-1c5a-7510123e34d2
md"## Bayesian Deep Learning

- [Uncertainty in Deep Learning](http://www.cs.ox.ac.uk/people/yarin.gal/website/blog_2248.html)
- [Bayesian Deep Learning 101](http://www.cs.ox.ac.uk/people/yarin.gal/website/bdl101/)
- [MLSS 2019 Skoltech Tutorials](https://github.com/mlss-skoltech/tutorials)

### Approximate Inference in Bayesian Neural Network with Dropout

> Drawing a new function for each test point makes no difference if all we care about is obtaining the predictive mean and predictive variance
> (actually, for these two quantities this process is preferable to the one I will describe below),
> but this process does not result in draws from the induced distribution over functions.
> ([Uncertainty in Deep Learning](http://www.cs.ox.ac.uk/people/yarin.gal/website/blog_2248.html))"
   

# ╔═╡ 279ea616-6ac1-11eb-10b3-ff31e5c3305e
begin
	bayesian_vgg11 = deepcopy(vgg11)
	trainmode!(bayesian_vgg11)
end

# ╔═╡ 92607376-6ac1-11eb-3620-1ff033ef6890
function point_estimate(model, X; n_samples=128)
	trainmode!(bayesian_vgg11)
	outputs = reduce(hcat, [Neural.predict(bayesian_vgg11, X) for i in 1:n_samples])
	ŷ_mean = dropdims(mean(outputs, dims=2), dims=2)
	ŷ_std = dropdims(std(outputs, mean=ŷ_mean, dims=2), dims=2)
	return ŷ_mean, ŷ_std
end

# ╔═╡ 77682b6c-6ac2-11eb-2aa3-01e02dea681e
ŷ_bayes, σ_bayes = point_estimate(bayesian_vgg11, X) |> cpu

# ╔═╡ 67316c8a-6ac3-11eb-0f13-538e1bb0b106
Evaluation.rmse(y, ŷ_bayes)

# ╔═╡ a352dbb8-6ac3-11eb-211a-8394d366557c
Evaluation.catastrophic_redshift_ratio(y, ŷ_bayes)

# ╔═╡ a8c710e6-6ac8-11eb-3f00-5d2b72864754
ŷ_error = y - ŷ_bayes

# ╔═╡ 2665ee04-6aca-11eb-29de-35e58a146d5c
cor(abs.(ŷ_error), σ_bayes)

# ╔═╡ 7b475aa2-6aca-11eb-2b1c-430dab75a520
argmax(ŷ_error), argmax(σ_bayes)

# ╔═╡ f3e971ec-6acb-11eb-28ca-717f2553512b
begin
	i_error_max = argmax(ŷ_error)
	Utils.plot_spectrum(X[:, i_error_max], label="error = $(ŷ_error[i_error_max])")
	Utils.plot_spectral_lines!(y[i_error_max])
	Utils.plot_spectral_lines!(ŷ_vgg11[i_error_max], color=:red, location=:bottom)
end

# ╔═╡ 84b92de8-6acc-11eb-041f-c1f730e5195f
begin
	i_σ_max = argmax(σ_bayes)
	Utils.plot_spectrum(X[:, i_σ_max], label="σ = $(σ_bayes[i_error_max])")
	Utils.plot_spectral_lines!(y[i_σ_max])
	Utils.plot_spectral_lines!(ŷ_vgg11[i_σ_max], color=:red, location=:bottom)
end

# ╔═╡ 92142f12-6acd-11eb-32c9-9f33783e75f4
σ_range = 0:0.01:maximum(σ_bayes)

# ╔═╡ c93ef038-6acc-11eb-2e6e-298f7178eb89
plot(
	σ_range,
	[Evaluation.rmse(y[σ_bayes .< σ], ŷ_bayes[σ_bayes .< σ]) for σ in σ_range],
	ylabel="RMSE")

# ╔═╡ 9f882f9c-6acd-11eb-1795-b7ec4224cf83
plot(
	σ_range,
	[Evaluation.catastrophic_redshift_ratio(y[σ_bayes .< σ], ŷ_bayes[σ_bayes .< σ])
		for σ in σ_range],
	ylabel="Catastrophic z Ratio")

# ╔═╡ bf58926c-6acd-11eb-3c23-bde13af4bfc2
plot(
	σ_range,
	[sum(σ_bayes .< σ) / length(σ_bayes) for σ in σ_range],
	ylabel="Completeness")

# ╔═╡ Cell order:
# ╟─ed4d438e-6aaa-11eb-051e-efe644cce631
# ╠═6dc764b2-66e7-11eb-0833-9dc54a18f920
# ╠═8643602a-66e9-11eb-3700-374047551428
# ╠═edd6e898-6797-11eb-2cee-791764fb425a
# ╟─f614d7c0-6aaa-11eb-1c62-73c54a75a0ab
# ╠═9e6ee084-6798-11eb-3683-b5c8c998cd77
# ╠═06de9ac0-679d-11eb-17d8-6f5359328d71
# ╠═67e19e7e-679f-11eb-0eb1-c39fc5eac3ed
# ╠═33c2836a-6aab-11eb-1c93-257ed54b0585
# ╟─0348b7e0-6aab-11eb-16ba-d574e6296961
# ╠═7b0e34c6-67ba-11eb-2500-378603362df8
# ╠═89f370c6-67ba-11eb-24da-2b0a0f0ad532
# ╠═98d6ed82-6ac3-11eb-0851-89add7ad3b38
# ╠═8d836ffc-67ba-11eb-0d82-f70af20aeeae
# ╠═1d52576a-6abb-11eb-2dd8-c388b2365ddd
# ╠═e6d5d6fa-6aab-11eb-0434-19c6a5a97099
# ╟─6e379afc-6ac0-11eb-1c5a-7510123e34d2
# ╠═279ea616-6ac1-11eb-10b3-ff31e5c3305e
# ╠═92607376-6ac1-11eb-3620-1ff033ef6890
# ╠═77682b6c-6ac2-11eb-2aa3-01e02dea681e
# ╠═67316c8a-6ac3-11eb-0f13-538e1bb0b106
# ╠═a352dbb8-6ac3-11eb-211a-8394d366557c
# ╠═a8c710e6-6ac8-11eb-3f00-5d2b72864754
# ╠═2665ee04-6aca-11eb-29de-35e58a146d5c
# ╠═7b475aa2-6aca-11eb-2b1c-430dab75a520
# ╠═f3e971ec-6acb-11eb-28ca-717f2553512b
# ╠═84b92de8-6acc-11eb-041f-c1f730e5195f
# ╠═92142f12-6acd-11eb-32c9-9f33783e75f4
# ╠═c93ef038-6acc-11eb-2e6e-298f7178eb89
# ╠═9f882f9c-6acd-11eb-1795-b7ec4224cf83
# ╠═bf58926c-6acd-11eb-3c23-bde13af4bfc2
