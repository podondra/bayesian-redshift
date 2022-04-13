### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ bf29119e-95b6-4988-b035-ea840b4c038b
using Distributions, Flux, Plots, Random, Zygote

# ╔═╡ a583e544-baf4-11ec-08c2-0bf6a3a428a7
begin
	n_sample = 2500
	y = Float32.(rand(Uniform(-10.5, 10.5), (1, n_sample)))
	ϵ = rand(Float32, (1, n_sample))
	X = sin.(0.75f0 * y) * 7.0f0 + y * 0.5f0 + ϵ
	scatter(transpose(X), transpose(y), alpha=0.3, label="training data")
end

# ╔═╡ 34480091-72ab-4497-b7c5-e39f510d9e25
md"# Fully Connected Neural Network"

# ╔═╡ dee49cf5-18e1-42c5-a137-f82dbcbaca4d
begin
	model = Chain(Dense(1, 20, tanh), Dense(20, 1))
	loss(x, y) = Flux.Losses.mse(model(x), y)
	optimizer = RMSProp(0.1, 0.8)
	parameters = Flux.params(model)
	for epoch in 1:1000
		Flux.train!(loss, parameters, [(X, y)], optimizer)
	end
end

# ╔═╡ 9bcb16fb-7b8e-4e7d-8600-44a340e6d9d1
begin
	X_test = -10.5f0:0.1f0:10.5f0
	X_test = reshape(X_test, 1, length(X_test))
	scatter(transpose(X), transpose(y), alpha=0.3, label="training data")
	scatter!(transpose(X_test), transpose(model(X_test)), label="test data") 
end

# ╔═╡ a2325767-9883-4232-a9f2-fafc86a19168
md"# Mixture Density Network"

# ╔═╡ cbdd3efb-328d-4735-b091-73a1f220298f
function normal(x, μ, σ)
	(1.0f0 ./ (σ * sqrt(2.0f0π))) .* exp.(-0.5f0 * ((x .- μ) ./ σ) .^ 2.0f0)
end

# ╔═╡ ecae47b1-30c0-4e95-ae8b-021b92f982a7
begin
	n_component = 24
	mdn = Chain(Dense(1, 24, tanh), Dense(24, 3 * n_component))
	
	function predict(mdn, X)
		z = mdn(X)
		return softmax(z[1:24, :]), z[25:48, :], exp.(z[49:72, :])
	end

	function loss_mdn(X, y)
		α, μ, σ = predict(mdn, X)
		return mean(-log.(sum(α .* normal(y, μ, σ), dims=1)))
	end

	opt = ADAM()
	ps = Flux.params(mdn)
	n_epoch = 10000
	losses = zeros(Float32, n_epoch)
	for epoch in 1:n_epoch
		train_loss, back = Zygote.pullback(() -> loss_mdn(X, y), ps)
		losses[epoch] = train_loss
		gs = back(one(train_loss))
		Flux.Optimise.update!(opt, ps, gs)
	end
	plot(1:n_epoch, losses, label="training loss")
end

# ╔═╡ 841d567b-4849-44df-bef0-d863eab2195e
begin
	α_test, μ_test, σ_test = predict(mdn, X_test)
	n_test = size(X_test, 2)
	y_test = zeros(Float32, n_test)
	for i in 1:n_test
		gmm = MixtureModel(
			Normal,
			[(μ, σ) for (μ, σ) in zip(μ_test[:, i], σ_test[:, i])],
			α_test[:, i])
		y_test[i] = rand(gmm)
	end
	
	scatter(transpose(X), transpose(y), alpha=0.3, label="training data")
	scatter!(transpose(X_test), y_test, label="sample")
	mode_test = μ_test[argmax(α_test, dims=1)]
	scatter!(transpose(X_test), transpose(mode_test), label="first mode")
	mean_test = sum(α_test .* μ_test, dims=1)
	scatter!(transpose(X_test), transpose(mean_test), label="mean")
end

# ╔═╡ Cell order:
# ╠═bf29119e-95b6-4988-b035-ea840b4c038b
# ╠═a583e544-baf4-11ec-08c2-0bf6a3a428a7
# ╟─34480091-72ab-4497-b7c5-e39f510d9e25
# ╠═dee49cf5-18e1-42c5-a137-f82dbcbaca4d
# ╠═9bcb16fb-7b8e-4e7d-8600-44a340e6d9d1
# ╟─a2325767-9883-4232-a9f2-fafc86a19168
# ╠═cbdd3efb-328d-4735-b091-73a1f220298f
# ╠═ecae47b1-30c0-4e95-ae8b-021b92f982a7
# ╠═841d567b-4849-44df-bef0-d863eab2195e
