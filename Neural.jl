module Neural

using BSON: @save
using CUDA
using Flux
using Flux.Data: DataLoader
using Flux.Losses: mse
using HDF5
using Logging
using TensorBoardLogger

export train_wrapper!, nn

function nn()
    Chain(
        Dense(512, 256, relu),
        Dropout(0.5),
        Dense(256, 128, relu),
        Dropout(0.5),
        Dense(128, 1))
end

function vgg11()
    Chain(
        Flux.unsqueeze(2),
        Conv((3, ), 1 => 64, relu, pad=1),
        MaxPool((2, )),
        Conv((3, ), 64 => 128, relu, pad=1),
        MaxPool((2, )),
        Conv((3, ), 128 => 256, relu, pad=1),
        Conv((3, ), 256 => 256, relu, pad=1),
        MaxPool((2, )),
        Conv((3, ), 256 => 512, relu, pad=1),
        Conv((3, ), 512 => 512, relu, pad=1),
        MaxPool((2, )),
        Conv((3, ), 512 => 512, relu, pad=1),
        Conv((3, ), 512 => 512, relu, pad=1),
        MaxPool((2, )),
        flatten,
        Dense(16 * 512, 4096, relu),
        Dropout(0.5),
        Dense(4096, 4096, relu),
        Dropout(0.5),
        Dense(4096, 1))
end

function get_data()
    datafile = h5open("data/dr16q_superset.hdf5")
    X_train = read(datafile, "X_tr")
    y_train = Flux.unsqueeze(convert(Array{Float32}, read(datafile, "z_tr")), 1)
    X_validation = read(datafile, "X_va")
    y_validation = Flux.unsqueeze(convert(Array{Float32}, read(datafile, "z_va")), 1)
    close(datafile)
    return X_train, y_train, X_validation, y_validation
end

function predict(model, X)
    reduce(hcat, [model(x) for x in DataLoader(X, batchsize=2048)])
end

function train_with_early_stopping!(
        model, X_train, y_train, X_validation, y_validation;
        batchsize, patience, device, file_model)
    model = model |> device
    X_train, y_train = X_train |> device, y_train |> device
    X_validation, y_validation = X_validation |> device, y_validation |> device

    loader_train = DataLoader(
        (X_train, y_train), batchsize=batchsize, shuffle=true)

    loss(x, y) = mse(model(x), y)

    # TODO weight decay
    optimizer = ADAM()
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
            @save file_model model
        else
            i += 1
        end
        @info "loss" validation=loss_validation train=loss_train
    end
end

function train_wrapper!(model, name_model)
    logger = TBLogger("runs/" * name_model, tb_overwrite)
    X_train, y_train, X_validation, y_validation = get_data()
    with_logger(logger) do
        train_with_early_stopping!(
            model, get_data()..., batchsize=256, patience=64, device=gpu,
            file_model=name_model * ".bson")
    end
end
end # module
