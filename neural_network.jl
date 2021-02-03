### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 90cd4814-6214-11eb-39bd-43e5f517eda8
begin
	using BSON: @load, @save
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
    y = convert(Array{Float32}, read(datafile, "z_tr"))
	X_va = read(datafile, "X_va")
    y_va = convert(Array{Float32}, read(datafile, "z_va"))
    close(datafile)
end

# ╔═╡ 35f4778a-6603-11eb-15ac-3bf999bb3241
function predict(model, X)
	predictions = Array{Float32}(undef, size(X, 2))
	dataloader = DataLoader(X, batchsize=1024)
	start = 1
	for xb in dataloader
		batchsize = size(xb, 2)
		predictions[start:start + batchsize - 1] = model(gpu(xb))
		start += batchsize
	end
	return predictions
end

# ╔═╡ f36cf690-65f7-11eb-087f-119bb1b29040
function train_early_stopping!(
		model, X, y, X_va, y_va, batchsize, patience, savepath)
    dataloader = DataLoader((X, y), batchsize=batchsize, shuffle=true)

    model = gpu(model)

    loss(x, y) = mse(model(x), y)
	losses, losses_va = Float32[], Float32[]

    # TODO weight decay
    opt = ADAM()
    ps = params(model)

    epoch = 1
    v_star = typemax(Float32)
	i = 0
    while i < patience
		trainmode!(model)
        for (xb, yb) in dataloader
            xb, yb = gpu(xb), gpu(yb)
            gs = gradient(ps) do
                loss(xb, yb)
            end
            update!(opt, ps, gs)
        end
        epoch += 1

		testmode!(model)
        v = mse(predict(model, X_va), y_va)
        if v < v_star
            i = 0
            v_star = v
			let model = cpu(model)
            	@save savepath model
			end
        else
            i += 1
        end
		push!(losses_va, v)
        push!(losses, mse(predict(model, X), y))
    end
	
	return losses, losses_va
end

# ╔═╡ 75a18a6a-6614-11eb-17c0-3d1ba4291412
function train_epochs!(model, X, y, X_va, y_va, batchsize, epochs)
    dataloader = DataLoader((X, y), batchsize=batchsize, shuffle=true)
	dataloader_va = DataLoader((X_va, y_va), batchsize=1024)
	
	model = gpu(model)
	
	loss(x, y) = mse(model(x), y)
	losses, losses_va = Float32[], Float32[]
	
	opt = ADAM()
	ps = Flux.params(model)
	
	for epoch in 1:epochs
		for (xb, yb) in dataloader
			xb, yb = gpu(xb), gpu(yb)
			
			gs = Flux.gradient(ps) do
				loss(xb, yb)
			end
			
			update!(opt, ps, gs)
		end
		
		validation_loss = 0f0
		for (xb, yb) in dataloader_va
			xb, yb = gpu(xb), gpu(yb)
			validation_loss += loss(xb, yb)
		end
		validation_loss /= length(dataloader_va)
		push!(losses_va, validation_loss)
	end
	
	return losses, losses_va
end

# ╔═╡ 3088ae56-6608-11eb-3a7b-3b10bbce7f18
begin
	nn = Chain(
		Dense(512, 256, relu), 
		Dropout(0.5), 
		Dense(256, 128, relu), 
		Dropout(0.5), 
		Dense(128, 1))

	losses, losses_va = train_epochs!(nn, X, y, X_va, y_va, 128, 32)
	
	plot(losses, label="training loss")
	plot!(losses_va, label="validation loss")
end

# ╔═╡ Cell order:
# ╠═90cd4814-6214-11eb-39bd-43e5f517eda8
# ╠═a4926ece-65f7-11eb-026d-b1c374bba55c
# ╠═35f4778a-6603-11eb-15ac-3bf999bb3241
# ╠═f36cf690-65f7-11eb-087f-119bb1b29040
# ╠═75a18a6a-6614-11eb-17c0-3d1ba4291412
# ╠═3088ae56-6608-11eb-3a7b-3b10bbce7f18
