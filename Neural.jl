module Neural

using BSON: bson
using CUDA
using Flux
using Flux.Data: DataLoader
using Flux.Losses: mse
using Flux.Optimise
using HDF5
using Logging
using NNlib
using TensorBoardLogger

export model, predict, train_wrapper!

function model()
    Chain(
        Flux.unsqueeze(2),
        Conv((9, ), 1 => 16, relu, pad=SamePad()),
        Dropout(0.5),
        MaxPool((2, )),
        Conv((9, ), 16 => 32, relu, pad=SamePad()),
        Dropout(0.5),
        MaxPool((2, )),
        flatten,
        Dense(4096, 1024, relu),
        Dropout(0.5),
        Dense(1024, 1024, relu),
        Dropout(0.5),
        Dense(1024, 1, relu))
end

function get_data()
    datafile = h5open("data/dr12q_superset.hdf5")
    X_train = read(datafile, "X_tr")
    y_train = read(datafile, "z_vi_tr")
    X_validation = read(datafile, "X_va")
    y_validation = read(datafile, "z_vi_va")
    close(datafile)
    return X_train, y_train, X_validation, y_validation
end

function predict(model, X)
    loader = DataLoader(X, batchsize=2048)
    reduce(vcat, [dropdims(model(x), dims=1) for x in loader])
end

function train_with_early_stopping!(
        model, X_train, y_train, X_validation, y_validation;
        batchsize, patience, weight_decay, device, file_model)
    model = model |> device
    X_train, y_train = X_train |> device, y_train |> device
    X_validation, y_validation = X_validation |> device, y_validation |> device

    loader_train = DataLoader(
        (X_train, y_train), batchsize=batchsize, shuffle=true)

    loss(x, y) = mse(dropdims(model(x), dims=1), y)

    optimizer = Optimiser(WeightDecay(weight_decay), ADAM())
    Θ = params(model)

    epoch = 1
    loss_validation_star = typemax(Float32)
    i = 0
    while i < patience
        Flux.train!(loss, Θ, loader_train, optimizer)
        epoch += 1

        loss_train = mse(predict(model, X_train), y_train)
        loss_validation = mse(predict(model, X_validation), y_validation)
        if loss_validation < loss_validation_star
            i = 0
            loss_validation_star = loss_validation
            bson(file_model, model=cpu(model))
        else
            i += 1
        end
        @info "loss" validation=loss_validation train=loss_train
    end
end

function train_wrapper!(model, name_model; bs=256, wd=1e-6)
    logger = TBLogger("runs/" * name_model, tb_overwrite)
    X_train, y_train, X_validation, y_validation = get_data()
    with_logger(logger) do
        train_with_early_stopping!(model, get_data()...,
            batchsize=bs, patience=32, weight_decay=wd,
            device=gpu, file_model="models/" * name_model * ".bson")
    end
end

end # module
