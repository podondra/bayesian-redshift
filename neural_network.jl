### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 90cd4814-6214-11eb-39bd-43e5f517eda8
begin
	using CUDA
	using Flux
	using Flux.Data: DataLoader
	using Flux.Losses: mse
	using Flux.Optimise: update!
	using HDF5
	using Plots
end

# ╔═╡ a4926ece-65f7-11eb-026d-b1c374bba55c
begin
    datafile = h5open("data/dr16q_superset.hdf5")
    X = read(datafile, "X_tr")
    y = Flux.unsqueeze(convert(Array{Float32}, read(datafile, "z_tr")), 1)
	X_va = read(datafile, "X_va")
    y_va = Flux.unsqueeze(convert(Array{Float32}, read(datafile, "z_va")), 1)
    close(datafile)
end

# ╔═╡ 21a9abc8-66bd-11eb-0f24-9f3dc6a10d78
size(X), size(y), size(X_va), size(y_va)

# ╔═╡ 232fd0da-66bd-11eb-31d1-433a72bfdfdd
typeof(X), typeof(y), typeof(X_va), typeof(y_va)

# ╔═╡ 35f4778a-6603-11eb-15ac-3bf999bb3241
function predict(model, X)
	loader = DataLoader(X, batchsize=2048)
	reduce(hcat, [model(x) for x in loader])
end

# ╔═╡ f36cf690-65f7-11eb-087f-119bb1b29040
function train_early_stopping!(
		model, X_train, y_train, X_validation, y_validation,
		batch_size, patience, device, file_model)
    model = model |> device
	X_train, y_train = X_train |> device, y_train |> device
	X_validation, y_validation = X_validation |> device, y_validation |> device

	loader_train = DataLoader(
		(X_train, y_train), batchsize=batch_size, shuffle=true)

    loss(x, y) = mse(model(x), y)
	losses_train, losses_validation = Float32[], Float32[]

    # TODO weight decay
    optimizer = ADAM()
    Θ = params(model)

    epoch = 1
    loss_validation_star = typemax(Float32)
	i = 0
	model_best = deepcopy(model)
    while i < patience
        Flux.train!(loss, Θ, loader_train, optimizer)
        epoch += 1

        loss_validation = mse(predict(model, X_validation), y_validation)
        if loss_validation < loss_validation_star
            i = 0
            loss_validation_star = loss_validation
			model_best = deepcopy(model)
        else
            i += 1
        end
		push!(losses_validation, loss_validation)
        push!(losses_train, mse(predict(model, X_train), y_train))
    end

	return model_best, losses_train, losses_validation
end

# ╔═╡ 3088ae56-6608-11eb-3a7b-3b10bbce7f18
begin
	nn = Chain(
		Dense(512, 256, relu), 
		Dropout(0.5), 
		Dense(256, 128, relu), 
		Dropout(0.5), 
		Dense(128, 1))

	nn_trained, losses_train, losses_valiation = train_early_stopping!(
		nn, X, y, X_va, y_va, 128, 16, gpu, "nn.bson")

	plot(losses_train, label="training loss")
	plot!(losses_valiation, label="validation loss")
end

# ╔═╡ Cell order:
# ╠═90cd4814-6214-11eb-39bd-43e5f517eda8
# ╠═a4926ece-65f7-11eb-026d-b1c374bba55c
# ╠═21a9abc8-66bd-11eb-0f24-9f3dc6a10d78
# ╠═232fd0da-66bd-11eb-31d1-433a72bfdfdd
# ╠═35f4778a-6603-11eb-15ac-3bf999bb3241
# ╠═f36cf690-65f7-11eb-087f-119bb1b29040
# ╠═3088ae56-6608-11eb-3a7b-3b10bbce7f18
