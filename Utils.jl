module Utils

using FITSIO
using Plots
using Printf

export LOGLAMMIN, LOGLAMMAX, get_filepath, get_spectrum, get_linear_spectrum,
       plot_spectral_lines!, plot_spectrum, polynomial_features

LOGLAMMIN, LOGLAMMAX = 3.5832, 3.9583
N_FEATURES = 3752
LAMBDA = 10 .^ range(LOGLAMMIN, LOGLAMMAX, length=N_FEATURES)

# http://classic.sdss.org/dr6/algorithms/linestable.html
LINES = Dict(1033.82 => "O VI",
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

function get_filename(plate, mjd, fiberid)
    @sprintf("%04d/spec-%04d-%05d-%04d.fits", plate, plate, mjd, fiberid)
end

function get_filepath(
        superset::String, plate::Int32, mjd::Int32, fiberid::Int32)::String
    @sprintf("data/%s/%04d/spec-%04d-%05d-%04d.fits",
             superset, plate, plate, mjd, fiberid)
end

function get_spectrum(
        superset::String, plate::Int32, mjd::Int32,
        fiberid::Int32)::Tuple{Vector{Float32}, Vector{Float32}}
    hdulist = FITS(Utils.get_filepath(superset, plate, mjd, fiberid))
    loglam = read(hdulist[2], "loglam")
    flux = read(hdulist[2], "flux")
    close(hdulist)
    return loglam, flux
end

function get_linear_spectrum(
        superset::String, plate::Int32, mjd::Int32,
        fiberid::Int32)::Tuple{Vector{Float32}, Vector{Float32}}
    hdulist = FITS(Utils.get_filepath(superset, plate, mjd, fiberid))
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
    plot(
        LAMBDA, flux;
        xlabel="Wavelength", ylabel="Flux", kwargs...)
end

function polynomial_features(x::Vector{Float32}, degree::Int64)::Matrix{Float64}
    X = Matrix{Float64}(undef, length(x), degree + 1)
    for col = 1:degree + 1
        X[:, col] = x .^ (col - 1)
    end
    return X
end

z2λ_obsv(z, λ_emit) = (1 + z) * λ_emit

end # module
