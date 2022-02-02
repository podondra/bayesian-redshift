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

export clf_fc, clf_sznet, reg_fc, reg_sznet,
       classify, mcdropout, regress,
       wrapper_reg!, wrapper_clf!, wrapper_mcdropout!

const WIDTH = 0.01f0
const LABELS = 0.0f0:WIDTH:6.44f0
const N_LABELS = length(LABELS)

function clf_fc()
    Chain(
        Dense(3752, 512, relu),
        Dropout(0.5),
        Dense(512, 512, relu),
        Dropout(0.5),
        Dense(512, N_LABELS))
end

function clf_sznet()
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

function reg_fc()
    Chain(
        Dense(3752, 512, relu),
        Dropout(0.5),
        Dense(512, 512, relu),
        Dropout(0.5),
        Dense(512, 1))
end

function reg_sznet()
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
    X_valid = read(datafile, "X_va")
    y_valid = read(datafile, "z_vi_va")
    close(datafile)

    return X_train, Flux.unsqueeze(y_train, 1), y_train, X_valid, y_valid
end

function get_clf_data()
    datafile = h5open("data/dr12q_superset.hdf5")
    X_train = read(datafile, "X_tr")
    y_train = read(datafile, "z_vi_tr")
    X_valid = read(datafile, "X_va")
    y_valid = read(datafile, "z_vi_va")
    close(datafile)

    EDGES = -0.005f0:WIDTH:6.445f0
    STR_LABELS = [@sprintf "%.2f" label for label in LABELS]
    y_train[y_train .< 0] .= 0    # z smaller than 0 should be zero
    y_train_categorical = cut(y_train, EDGES, labels=STR_LABELS)
    y_train_onehot = Flux.onehotbatch(y_train_categorical, STR_LABELS)

    return X_train, y_train_onehot, y_train, X_valid, y_valid
end

function forward(model, X)
    reduce(hcat, cpu([model(x) for x in DataLoader(X, batchsize=2048)]))
end

function regress(model, X)
    dropdims(forward(model, X), dims=1)
end

function classify(model, X)
    Flux.onecold(forward(model, X), LABELS)
end

function mcdropout(model, X; T=20)
    trainmode!(model)
    p = zeros(N_LABELS, size(X, 2))
    for i in 1:T
        p += softmax(forward(model, X))
    end
    return Flux.onecold(p / T, LABELS)
end

function train!(
        model, X_train, y_train_encoded, y_train, X_valid, y_valid;
        loss, predict, batchsize, patience, weight_decay, file_model)
    model = gpu(model)
    X_train, y_train_encoded = gpu(X_train), gpu(y_train_encoded)
    X_valid = gpu(X_valid)

    loader_train = DataLoader((X_train, y_train_encoded), batchsize=batchsize, shuffle=true)

    optimizer = Optimiser(WeightDecay(weight_decay), ADAM())
    Θ = params(model)

    loss_function(x, y) = loss(model(x), y)

    epoch = 1
    cfr_valid_star = typemax(Float32)
    i = 0
    while i < patience
        Flux.train!(loss_function, Θ, loader_train, optimizer)
        epoch += 1

        ŷ_valid = predict(model, X_valid)
        ŷ_train = predict(model, X_train)

        cfr_valid = cfr(y_valid, ŷ_valid)
        mean_Δv_valid = meanΔv(y_valid, ŷ_valid)
        median_Δv_valid = medianΔv(y_valid, ŷ_valid)
        rmse_valid = rmse(y_valid, ŷ_valid)

        cfr_train = cfr(y_train, ŷ_train)
        mean_Δv_train = meanΔv(y_train, ŷ_train)
        median_Δv_train = medianΔv(y_train, ŷ_train)
        rmse_train = rmse(y_train, ŷ_train)

        @info "catastrophic failure rate" log_step_increment=0 validation=cfr_valid train=cfr_train
        @info "mean Δv" log_step_increment=0 validation=mean_Δv_valid train=mean_Δv_train
        @info "median Δv" log_step_increment=0 validation=median_Δv_valid train=median_Δv_train
        @info "rms error" validation=rmse_valid train=rmse_train

        if cfr_valid < cfr_valid_star
            i = 0
            cfr_valid_star = cfr_valid
            bson(file_model, model=cpu(model))
        else
            i += 1
        end
    end
end

function wrapper_reg!(model, model_name; bs=256, wd=0)
    logger = TBLogger("runs/" * model_name)
    with_logger(logger) do
        train!(
            model, get_data()...,
            loss=mse, predict=regress,
            batchsize=bs, patience=32, weight_decay=wd,
            file_model="models/" * model_name * ".bson")
    end
end

function wrapper_clf!(model, model_name; bs=256, wd=0)
    logger = TBLogger("runs/" * model_name)
    with_logger(logger) do
        train!(
            model, get_clf_data()...,
            loss=logitcrossentropy, predict=classify,
            batchsize=bs, patience=32, weight_decay=wd,
            file_model="models/" * model_name * ".bson")
    end
end

function wrapper_mcdropout!(model, model_name; bs=256, wd=1e-4)
    logger = TBLogger("runs/" * model_name)
    with_logger(logger) do
        train!(
            model, get_clf_data()...,
            loss=logitcrossentropy, predict=mcdropout,
            batchsize=bs, patience=32, weight_decay=wd,
            file_model="models/" * model_name * ".bson")
    end
end

end # module
