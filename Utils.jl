module Utils

using FITSIO
using Plots
using Printf

export get_filepath, get_spectrum, plot_spectrum, plot_spectral_lines!

function get_filepath(plate::Int32, mjd::Int32, fiberid::Int32)::String
    @sprintf("data/DR16Q_Superset_v3/%04d/spec-%04d-%05d-%04d.fits",
             plate, plate, mjd, fiberid)
end

function get_spectrum(
        plate::Int32,
        mjd::Int32,
        fiberid::Int32)::Tuple{Vector{Float32}, Vector{Float32}}
    filepath = get_filepath(plate, mjd, fiberid)
    hdulist = FITS(Utils.get_filepath(plate, mjd, fiberid))
    loglam = read(hdulist[2], "loglam")
    flux = read(hdulist[2], "flux")
    close(hdulist)
    return loglam, flux
end

EPS = 0.0005
LOGLAMMIN, LOGLAMMAX = 3.5812 + EPS, 3.9637 - EPS
N_FEATURES = 512
LAMBDA = 10 .^ range(LOGLAMMIN, LOGLAMMAX, length=N_FEATURES)

function plot_spectrum(flux; kwargs...)
    plot(
        LAMBDA, flux;
        xlabel="Wavelength", ylabel="Flux", kwargs...)
end

# http://classic.sdss.org/dr6/algorithms/linestable.html
LINES = Dict("Lyα" => 1215.24,
             "C IV" => 1549.48,
             "C III" => 1908.734,
             "Mg II" => 2799.117,
             "O III" => 1665.85,
             "Hα" => 6564.61)

z2λ_obsv(z, λ_emit) = (1 + z) * λ_emit

function plot_spectral_lines!(z; color=:black, location=:top)
    scatter!(
        [z2λ_obsv(z, λ_emit) for λ_emit in values(LINES)],
        zeros(length(values(LINES))),
        xlim=(LAMBDA[1], LAMBDA[end]),
        marker=(:vline, color), label=:none,
        series_annotation=[text(line, 8, location) for line in keys(LINES)])
end

end # module
