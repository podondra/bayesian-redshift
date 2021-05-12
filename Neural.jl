module Neural

using BSON: bson
using CategoricalArrays
using CUDA
using Flux
using Flux.Data: DataLoader
using Flux.Losses: logitcrossentropy, mse
using Flux.Optimise
using HDF5
using Logging
using NNlib
using Printf
using Statistics
using TensorBoardLogger

include("Evaluation.jl")
using .Evaluation

export bayesian_model, regression_model, classification_model, classify, regress,
    train_wrapper_regression!, train_wrapper_classification!

function classification_model()
    Chain(
        Flux.unsqueeze(2),
        Conv((3, ), 1 => 4, relu, pad=SamePad()),
        Conv((3, ), 4 => 8, relu, pad=SamePad()),
        flatten,
        Dense(2048, 2048, relu),
        Dropout(0.5),
        Dense(2048, 2048, relu),
        Dropout(0.5),
        Dense(2048, 599))
end

function regression_model()
    Chain(
        Flux.unsqueeze(2),
        Conv((3, ), 1 => 4, relu, pad=SamePad()),
        Conv((3, ), 4 => 8, relu, pad=SamePad()),
        flatten,
        Dense(2048, 2048, relu),
        Dense(2048, 2048, relu),
        Dense(2048, 1))
end

function get_data()
    datafile = h5open("data/dr12q_superset.hdf5")
    X_train = read(datafile, "X_tr")
    y_train = read(datafile, "z_vi_tr")
    X_validation = read(datafile, "X_va")
    y_validation = read(datafile, "z_vi_va")
    close(datafile)
    return X_train, Flux.unsqueeze(y_train, 1), y_train, X_validation, y_validation
end

function get_classification_data()
    datafile = h5open("data/dr12q_superset.hdf5")
    X_train = read(datafile, "X_tr")
    y_train = read(datafile, "z_vi_tr")
    X_validation = read(datafile, "X_va")
    y_validation = read(datafile, "z_vi_va")
    close(datafile)

    EDGES = -0.005f0:0.01f0:5.985f0
    LABELS = [@sprintf "%.2f" label for label in 0:0.01:5.98]
    y_train_categorical = cut(y_train, EDGES, labels=LABELS)
    y_train_onehot = Flux.onehotbatch(y_train_categorical, LABELS)

    return X_train, y_train_onehot, y_train, X_validation, y_validation
end

function regress(model, X)
    loader = DataLoader(X, batchsize=2048)
    return dropdims(reduce(hcat, [model(x) for x in loader]), dims=1)
end

function classify(model, X)
    loader = DataLoader(X, batchsize=2048)
    return reduce(vcat, [Flux.onecold(model(x), 0.00f0:0.01f0:5.98f0) for x in loader])
end

function train_with_early_stopping!(
        model, X_train, y_train_encoded, y_train, X_validation, y_validation;
        loss, predict, batchsize, patience, weight_decay, device, file_model)
    model = model |> device
    X_train, y_train_encoded = X_train |> device, y_train_encoded |> device
    X_validation = X_validation |> device

    loader_train = DataLoader(
        (X_train, y_train_encoded), batchsize=batchsize, shuffle=true)

    optimizer = Optimiser(WeightDecay(weight_decay), ADAM())
    Θ = params(model)

    loss_function(x, y) = loss(model(x), y)

    epoch = 1
    cat_z_validation_star = typemax(Float32)
    i = 0
    while i < patience
        Flux.train!(loss_function, Θ, loader_train, optimizer)
        epoch += 1

        ŷ_train = predict(model, X_train) |> cpu
        ŷ_validation = predict(model, X_validation) |> cpu

        rmse_train = rmse(y_train, ŷ_train)
        rmse_validation = rmse(y_validation, ŷ_validation)
        @info "rmse" train=rmse_train validation=rmse_validation
        cat_z_train = cat_z_ratio(y_train, ŷ_train)
        cat_z_validation = cat_z_ratio(y_validation, ŷ_validation)
        @info "catastrophic z ratio" train=cat_z_train validation=cat_z_validation

        if cat_z_validation < cat_z_validation_star
            i = 0
            cat_z_validation_star = cat_z_validation
            bson(file_model, model=cpu(model))
        else
            i += 1
        end
    end
end

function train_wrapper_regression!(model, model_name; bs=256, wd=0)
    logger = TBLogger("runs/" * model_name, tb_overwrite)
    with_logger(logger) do
        train_with_early_stopping!(
            model, get_data()...,
            loss=mse, predict=regress,
            batchsize=bs, patience=32, weight_decay=wd, device=gpu,
            file_model="models/" * model_name * ".bson")
    end
end

function train_wrapper_classification!(model, model_name; bs=256, wd=0)
    logger = TBLogger("runs/" * model_name, tb_overwrite)
    with_logger(logger) do
        train_with_early_stopping!(
            model, get_classification_data()...,
            loss=logitcrossentropy, predict=classify,
            batchsize=bs, patience=32, weight_decay=wd, device=gpu,
            file_model="models/" * model_name * ".bson")
    end
end

end # module
