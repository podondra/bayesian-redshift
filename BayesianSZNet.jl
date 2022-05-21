__precompile__()    # this module is safe to precompile
module BayesianSZNet

using BSON
using DelimitedFiles
using Distributions
using FITSIO
using Flux, Flux.Data, Flux.Losses, Flux.Optimise
using HDF5
using Logging
using Plots
using Printf
using PyCall
using Statistics
using StatsBase
using TensorBoardLogger

export computeΔv, mcrps, rmse, pithist
export FCNN, SZNet
export getdr12data, forward, multitrain, sample, train!
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
const ps = PyNULL()

function getdr12data()
    datafile = h5open("data/dr12q_superset.hdf5")
    X_tr, z_tr = read(datafile, "X_tr"), read(datafile, "z_vi_tr")
    X_va, z_va = read(datafile, "X_va"), read(datafile, "z_vi_va")
    close(datafile)
    return X_tr, Flux.unsqueeze(z_tr, 1), X_va, Flux.unsqueeze(z_va, 1)
end

# https://github.com/JuliaPy/PyCall.jl#using-pycall-from-julia-modules
function __init__()
    copy!(ps, pyimport("properscoring"))
end

function mcrps(z, ẑs)
    return mean(ps.crps_ensemble(dropdims(z, dims=1), transpose(ẑs)))
end

function rmse(z, ẑ)
    difference = dropdims(z - ẑ, dims=1)
    return sqrt((1 / length(difference)) * difference' * difference)
end

function pithist(z, ẑs)
    histogram(
        [ecdf(ẑs[:, i])(z[i]) for i in 1:length(z)], normalize=:pdf,
        xlabel="Probability Integral Transform", ylabel="Density", label=:none)
end

computeΔv(z, ẑ) = C .* ((ẑ - z) ./ (1 .+ z))

abstract type Model end

struct FCNN <: Model model end
FCNN(modelfile::String) = FCNN(gpu(trainmode!(BSON.load(modelfile, @__MODULE__)[:model])))
FCNN(p::AbstractFloat) = FCNN(gpu(trainmode!(Chain(Dense(3752, 512, relu),
                                                   Dropout(p),
                                                   Dense(512, 512, relu),
                                                   Dropout(p),
                                                   Dense(512, 1)))))

struct SZNet <: Model model end
SZNet(modelfile::String) = SZNet(gpu(trainmode!(BSON.load(modelfile, @__MODULE__)[:model])))
SZNet(p::AbstractFloat) = SZNet(gpu(trainmode!(Chain(Flux.unsqueeze(2),
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
                                                     Dense(512, 1)))))

function forward(model::Model, X; batchsize=1024)
    return cpu(reduce(hcat, [model.model(x) for x in DataLoader(X, batchsize=batchsize)]))
end

function sample(model::Model, X; batchsize=1024, T=256)
    return reduce(vcat, [forward(model, X, batchsize=batchsize) for t in 1:T])
end

function train!(model, X_tr, z_tr, X_va, z_va; modelname, patience, λ)
    X_tr_gpu, z_tr_gpu, X_va_gpu = gpu(X_tr), gpu(z_tr), gpu(X_va)
    loader = DataLoader((X_tr_gpu, z_tr_gpu), batchsize=256, shuffle=true)
    optimizer = Optimiser(WeightDecay(λ), ADAM())
    Θ = Flux.params(model.model)
    loss_function(x, z) = mse(model.model(x), z)
    mcrps_va_star = typemax(Float32)
    with_logger(TBLogger("runs/" * modelname, tb_overwrite)) do
        i = 0
        while i < patience
            Flux.train!(loss_function, Θ, loader, optimizer)
            ẑs_va = sample(model, X_va_gpu)
            # mean continuous rank probability score (MCRPS)
            mcrps_va = mcrps(z_va, ẑs_va)
            @info "mcrps" validation=mcrps_va
            if mcrps_va < mcrps_va_star
                i = 0
                mcrps_va_star = mcrps_va
                bson("models/" * modelname * ".bson", model=cpu(model.model))    # save model
            else
                i += 1
            end
        end
    end
    return mcrps_va_star
end

function multitrain(Constructor, X_tr, z_tr, X_va, z_va; constructorname, p, patience, runs=10, λ)
    for i in 1:runs
        modelname = "$(constructorname)_$(i)_$(p)_$(λ)"
        model = Constructor(p)
        mcrps_va_star = train!(model, X_tr, z_tr, X_va, z_va; modelname, patience, λ)
        println(mcrps_va_star)
    end
end

function confint(x; α=0.05)
    n = length(x)
    return cquantile(TDist(n - 1), α / 2) * std(x) / sqrt(n)
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
