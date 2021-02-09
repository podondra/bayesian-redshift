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
	y = convert(Array{Float32}, read(datafile, "z_va")) |> gpu
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
ŷ_nn = Neural.predict(nn, X)

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
ŷ_vgg11 = Neural.predict(vgg11, X)

# ╔═╡ 89f370c6-67ba-11eb-24da-2b0a0f0ad532
Evaluation.rmse(y, ŷ_vgg11)

# ╔═╡ 8d836ffc-67ba-11eb-0d82-f70af20aeeae
Evaluation.catastrophic_redshift_ratio(y, ŷ_vgg11, threshold=6000)

# ╔═╡ 1d52576a-6abb-11eb-2dd8-c388b2365ddd
begin
	rnd_i = rand(1:size(id, 1))
	loglam, flux = Utils.get_spectrum(id[rnd_i, :]...)
	plot(10 .^ loglam, flux)
	#Utils.plot_spectrum(X[:, rnd_i])
	Utils.plot_spectral_lines!(y[rnd_i])
	Utils.plot_spectral_lines!(ŷ_vgg11[rnd_i], color=:red, location=:bottom)
end

# ╔═╡ e6d5d6fa-6aab-11eb-0434-19c6a5a97099
begin
	density(y, label="Validation Set", xlabel="z", ylabel="Density")
	density!(ŷ_vgg11, label="VGG11 Predictions")
end

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
# ╠═8d836ffc-67ba-11eb-0d82-f70af20aeeae
# ╠═1d52576a-6abb-11eb-2dd8-c388b2365ddd
# ╠═e6d5d6fa-6aab-11eb-0434-19c6a5a97099
