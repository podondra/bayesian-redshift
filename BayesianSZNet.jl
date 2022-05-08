module BayesianSZNet

using BSON
using DelimitedFiles
using FITSIO
using Flux, Flux.Data, Flux.Losses, Flux.Optimise
using HDF5
using Logging
using Plots
using Printf
using Statistics
using TensorBoardLogger

export rmse, computeΔv
export FCNN, SZNet
export getdr12data, predict, train!
export LOGLAMMIN, LOGLAMMAX
export get_filename, get_filepath, get_spectrum, get_linear_spectrum
export writelst
export plot_spectral_lines!, plot_spectrum
export polynomial_features

const C = 299792.458    # the speed of light in vacuum (km / s)
const LOGLAMMIN, LOGLAMMAX = 3.5832, 3.9583
const N_FEATURES = 3752
const LAMBDA = 10 .^ range(LOGLAMMIN, LOGLAMMAX, length=N_FEATURES)
# http://classic.sdss.org/dr6/algorithms/linestable.html
const LINES = Dict(1033.82 => "O VI",
                   1215.24 => "Lyα",
                   1240.81 => "N V",
                   1399. => "Si IV + O IV",
                   1549.48 => "C IV",
                   1908.734 => "C III",
                   2326.0 => "C II",
                   2799.117 => "Mg II",
                   3727.092 => "O II",
                   3934.777 => "K",
                   3969.588 => "H",
                   4102.89 =>"HΔ",
                   4305.61 => "G",
                   4341.68 => "Hγ",
                   4862.68 => "Hβ",
                   4960.295 => "O III",
                   5008.240 => "O III",
                   5176.7 => "Mg",
                   5895.6 => "Na",
                   #6549.86 => "N II",
                   6564.61 => "Hα",
                   #6585.27 => "N II",
                   #6718.29 => "S II",
                   #6732.67 => "S II",
                   8500.36 => "Ca II",
                   8544.44 => "Ca II",
                   8664.52 => "Ca II")


function getdr12data()
    datafile = h5open("data/dr12q_superset.hdf5")
    X_tr, z_tr = read(datafile, "X_tr"), read(datafile, "z_vi_tr")
    X_va, z_va = read(datafile, "X_va"), read(datafile, "z_vi_va")
    close(datafile)
    return X_tr, Flux.unsqueeze(z_tr, 1), X_va, Flux.unsqueeze(z_va, 1)
end

function rmse(z, ẑ)
    difference = dropdims(z - ẑ, dims=1)
    sqrt((1 / length(difference)) * difference' * difference)
end

computeΔv(z, ẑ) = C .* ((ẑ - z) ./ (1 .+ z))

abstract type Model end

struct FCNN <: Model model end
FCNN(modelfile::String) = FCNN(gpu(BSON.load(modelfile, @__MODULE__)[:model]))
FCNN(p::AbstractFloat) = FCNN(gpu(Chain(Dense(3752, 512, relu),
                                        Dropout(p),
                                        Dense(512, 512, relu),
                                        Dropout(p),
                                        Dense(512, 512, relu),
                                        Dropout(p),
                                        Dense(512, 512, relu),
                                        Dropout(p),
                                        Dense(512, 1))))

struct SZNet <: Model model end
SZNet(modelfile::String) = SZNet(gpu(BSON.load(modelfile, @__MODULE__)[:model]))
SZNet(p::AbstractFloat) = SZNet(gpu(Chain(Flux.unsqueeze(2),
                                          Conv((3,), 1 => 8, relu, pad=SamePad()),
                                          MaxPool((2,)),
                                          Conv((3,), 8 => 16, relu, pad=SamePad()),
                                          MaxPool((2,)),
                                          Conv((3,), 16 => 32, relu, pad=SamePad()),
                                          Conv((3,), 32 => 32, relu, pad=SamePad()),
                                          MaxPool((2,)),
                                          Conv((3,), 32 => 64, relu, pad=SamePad()),
                                          Conv((3,), 64 => 64, relu, pad=SamePad()),
                                          MaxPool((2,)),
                                          Conv((3,), 64 => 64, relu, pad=SamePad()),
                                          Conv((3,), 64 => 64, relu, pad=SamePad()), 
                                          MaxPool((2,)),
                                          Flux.flatten,
                                          Dense(7488, 512, relu),
                                          Dropout(p),
                                          Dense(512, 512, relu),
                                          Dropout(p),
                                          Dense(512, 1))))

function predict(model::Model, X; batchsize=2048)
    return cpu(reduce(hcat, [model.model(x) for x in DataLoader(X, batchsize=batchsize)]))
end

function train!(model, X_tr, z_tr, X_va, z_va; modelname, patience=256, weightdecay=0.0)
    X_tr_gpu, z_tr_gpu, X_va_gpu = gpu(X_tr), gpu(z_tr), gpu(X_va)
    loader = DataLoader((X_tr_gpu, z_tr_gpu), batchsize=256, shuffle=true)
    optimizer = Optimiser(WeightDecay(weightdecay), ADAM())
    Θ = Flux.params(model.model)
    loss_function(x, z) = mse(model.model(x), z)
    with_logger(TBLogger("runs/" * modelname, tb_overwrite)) do
        epoch = 1
        rmse_va_star = typemax(Float32)
        i = 0
        while i < patience
            Flux.train!(loss_function, Θ, loader, optimizer)
            epoch += 1
            rmse_tr = rmse(z_tr, predict(model, X_tr_gpu))
            rmse_va = rmse(z_va, predict(model, X_va_gpu))
            @info "rmse" train=rmse_tr validation=rmse_va
            if rmse_va < rmse_va_star
                i = 0
                rmse_va_star = rmse_va
                bson("models/" * modelname * ".bson", model=cpu(model.model))    # save model
            else
                i += 1
            end
        end
    end
end

function get_filename(plate, mjd, fiberid)
    @sprintf("%04d/spec-%04d-%05d-%04d.fits", plate, plate, mjd, fiberid)
end

function get_filepath(superset::String, plate::Int32, mjd::Int32, fiberid::Int32)::String
    @sprintf("data/%s/%04d/spec-%04d-%05d-%04d.fits", superset, plate, plate, mjd, fiberid)
end

function get_spectrum(
        superset::String, plate::Int32, mjd::Int32,
        fiberid::Int32)::Tuple{Vector{Float32}, Vector{Float32}}
    hdulist = FITS(get_filepath(superset, plate, mjd, fiberid))
    loglam = read(hdulist[2], "loglam")
    flux = read(hdulist[2], "flux")
    close(hdulist)
    return loglam, flux
end

function get_linear_spectrum(
        superset::String, plate::Int32, mjd::Int32,
        fiberid::Int32)::Tuple{Vector{Float32}, Vector{Float32}}
    hdulist = FITS(get_filepath(superset, plate, mjd, fiberid))
    lam = 10 .^ read(hdulist[2], "loglam")
    flux = read(hdulist[2], "flux")
    close(hdulist)
    return lam, flux
end

function plot_spectral_lines!(z; color=:black, location=:top)
    scatter!(
        [z2λ_obsv(z, λ_emit) for λ_emit in keys(LINES)],
        zeros(length(keys(LINES))),
        xlim=(LAMBDA[1], LAMBDA[end]),
        marker=(:vline, color), label=:none,
        series_annotation=[text(line, 8, location) for line in values(LINES)])
end

function plot_spectrum(flux; kwargs...)
    plot(LAMBDA, flux; xlabel="Wavelength", ylabel="Flux", kwargs...)
end

function polynomial_features(x::Vector{Float32}, degree::Int64)::Matrix{Float64}
    X = Matrix{Float64}(undef, length(x), degree + 1)
    for col = 1:degree + 1
        X[:, col] = x .^ (col - 1)
    end
    return X
end

z2λ_obsv(z, λ_emit) = (1 + z) * λ_emit

function writelst(filepath, plates, mjds, fiberids)
    open(filepath, "w") do file
        writedlm(file, get_filename.(plates, mjds, fiberids))
    end
end 

end
