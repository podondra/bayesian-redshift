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

export classification_fc, classification_model, regression_model, regression_fc,
       classify, mc_dropout, regress,
       train_wrapper_regression!, train_wrapper_classification!, train_wrapper_mc_dropout!

const LABELS = 0.00f0:0.01f0:6.44f0
const N_LABELS = length(LABELS)

function classification_fc()
    Chain(
        Dense(3752, 512, relu),
        Dropout(0.5),
        Dense(512, 512, relu),
        Dropout(0.5),
        Dense(512, N_LABELS))
end

function classification_model()
    Chain(
        Flux.unsqueeze(2),
        Conv((3,), 1=>8, relu, pad=SamePad()),
        MaxPool((2,)),
        Conv((3,), 8=>16, relu, pad=SamePad()),
        MaxPool((2,)),
        Conv((3,), 16=>32, relu, pad=SamePad()),
        Conv((3,), 32=>32, relu, pad=SamePad()),
        MaxPool((2,)),
        Conv((3,), 32=>64, relu, pad=SamePad()),
        Conv((3,), 64=>64, relu, pad=SamePad()),
        MaxPool((2,)),
        Conv((3,), 64=>64, relu, pad=SamePad()),
        Conv((3,), 64=>64, relu, pad=SamePad()),
        MaxPool((2,)),
        flatten,
        Dense(7488, 512, relu),
        Dropout(0.5),
        Dense(512, 512, relu),
        Dropout(0.5),
        Dense(512, N_LABELS))
end

function regression_fc()
    Chain(
        Dense(3752, 512, relu),
        Dropout(0.5),
        Dense(512, 512, relu),
        Dropout(0.5),
        Dense(512, 1))
end

function regression_model()
    Chain(
        Flux.unsqueeze(2),
        Conv((3,), 1=>8, relu, pad=SamePad()),
        MaxPool((2,)),
        Conv((3,), 8=>16, relu, pad=SamePad()),
        MaxPool((2,)),
        Conv((3,), 16=>32, relu, pad=SamePad()),
        Conv((3,), 32=>32, relu, pad=SamePad()),
        MaxPool((2,)),
        Conv((3,), 32=>64, relu, pad=SamePad()),
        Conv((3,), 64=>64, relu, pad=SamePad()),
        MaxPool((2,)),
        Conv((3,), 64=>64, relu, pad=SamePad()),
        Conv((3,), 64=>64, relu, pad=SamePad()),
        MaxPool((2,)),
        flatten,
        Dense(7488, 512, relu),
        Dropout(0.5),
        Dense(512, 512, relu),
        Dropout(0.5),
        Dense(512, 1))
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

    EDGES = -0.005f0:0.01f0:6.445f0
    STR_LABELS = [@sprintf "%.2f" label for label in 0:0.01:6.44]
    # z smaller than 0 should be zero
    y_train[y_train .< 0] .= 0
    y_train_categorical = cut(y_train, EDGES, labels=STR_LABELS)
    y_train_onehot = Flux.onehotbatch(y_train_categorical, STR_LABELS)

    return X_train, y_train_onehot, y_train, X_validation, y_validation
end

function forward_pass(model, X)
    reduce(hcat, cpu([model(x) for x in DataLoader(X, batchsize=2048)]))
end

function regress(model, X)
    dropdims(forward_pass(model, X), dims=1)
end

function classify(model, X)
    Flux.onecold(forward_pass(model, X), LABELS)
end

function mc_dropout(model, X; T=20)
    trainmode!(model)
    p = zeros(N_LABELS, size(X, 2))
    for i in 1:T
        p += softmax(forward_pass(model, X))
    end
    return Flux.onecold(p / T, LABELS)
end

function train_with_early_stopping!(
        model, X_train, y_train_encoded, y_train, X_validation, y_validation;
        loss, predict, batchsize, patience, weight_decay, file_model)
    model = gpu(model)
    X_train, y_train_encoded = gpu(X_train), gpu(y_train_encoded)
    X_validation = gpu(X_validation)

    loader_train = DataLoader((X_train, y_train_encoded), batchsize=batchsize, shuffle=true)

    optimizer = Optimiser(WeightDecay(weight_decay), ADAM())
    Θ = params(model)

    loss_function(x, y) = loss(model(x), y)

    epoch = 1
    cat_z_validation_star = typemax(Float32)
    i = 0
    while i < patience
        Flux.train!(loss_function, Θ, loader_train, optimizer)
        epoch += 1

        ŷ_validation = predict(model, X_validation)
        ŷ_train = predict(model, X_train)

        rmse_validation = rmse(y_validation, ŷ_validation)
        rmse_train = rmse(y_train, ŷ_train)
        @info "rmse" validation=rmse_validation train=rmse_train log_step_increment=0
        cat_z_validation = cat_z_ratio(y_validation, ŷ_validation)
        cat_z_train = cat_z_ratio(y_train, ŷ_train)
        @info "catastrophic z ratio" validation=cat_z_validation train=cat_z_train

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
            batchsize=bs, patience=32, weight_decay=wd,
            file_model="models/" * model_name * ".bson")
    end
end

function train_wrapper_classification!(model, model_name; bs=256, wd=0)
    logger = TBLogger("runs/" * model_name, tb_overwrite)
    with_logger(logger) do
        train_with_early_stopping!(
            model, get_classification_data()...,
            loss=logitcrossentropy, predict=classify,
            batchsize=bs, patience=32, weight_decay=wd,
            file_model="models/" * model_name * ".bson")
    end
end

function train_wrapper_mc_dropout!(model, model_name; bs=256, wd=1e-4)
    logger = TBLogger("runs/" * model_name, tb_overwrite)
    with_logger(logger) do
        train_with_early_stopping!(
            model, get_classification_data()...,
            loss=logitcrossentropy, predict=mc_dropout,
            batchsize=bs, patience=32, weight_decay=wd,
            file_model="models/" * model_name * ".bson")
    end
end

end # module
