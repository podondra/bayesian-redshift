using FITSIO, HDF5, LinearAlgebra, Printf

function polynomial_features(x::Vector{Float64}, degree::Int64)::Matrix{Float64}
	n = length(x)
	X = Matrix{Float64}(undef, n, degree + 1)
	for col = 1:degree + 1
		X[:, col] = x .^ (col - 1)
	end
	return X
end

function fit(X::Matrix{Float64}, y)::Vector{Float64}
    return inv(X' * X) * X' * y
end

function arrange_layer!(
		X::Matrix{Float64}, y::Vector{Float64}, a::Vector{Float64},
		n_current::Int64, width::Float64, removal_parameter::Float64)::Int64
    remove_distance = removal_parameter * width
    while true
        i = 1
        n_init = n_current
        while i <= n_current
            y_pred = dot(X[i, :], a)
            if abs(y[i] - y_pred) >= remove_distance
                y_pred = dot(X[n_current, :], a)
                if abs(y[n_current] - y_pred) < remove_distance
                    # swap
                    X[i, :], X[n_current, :] = X[n_current, :], X[i, :]
                    y[i], y[n_current] = y[n_current], y[i]
                end
                n_current -= 1
            else
                i += 1
            end
        end
        if (n_current == n_init) || (n_current <= length(a) + 3)
            break
        end
        a[:] = fit(X[1:n_current, :], y[1:n_current])
    end
    return n_current
end

function dls_fit(
		X::Matrix{Float64}, y::Vector{Float64},
		k::Int64, removal_parameter::Float64)
	n, m = size(X)
	subset_n, subset_dls = Int64[], Float64[]

	a = fit(X, y)
	while n > m + 3
		d = abs.(y[1:n] - X[1:n, :] * a)
		width = maximum(d)
		dls = sum(d .^ 2) / width .^ k

		push!(subset_n, n)
		push!(subset_dls, dls)

		n = arrange_layer!(X, y, a, n, width, removal_parameter)
	end

	n_best = subset_n[argmax(subset_dls)]
	a_best = fit(X[1:n_best, :], y[1:n_best])
	return n_best, a_best
end

const N_WAVES = 3826
const LOGLAMMIN = 3.5812 - 0.00005
const LOGLAMMAX = 3.9637 + 0.00005

fid_read = h5open("data/dr16q_superset.hdf5", "r")
ids = read(fid_read["id"])
n = length(ids)
close(fid_read)

k = 2
removal_parameter = 0.9
degree = 3
flux_dset = Matrix{Float32}(undef, n, N_WAVES)

for i = 1:n
	id = ids[i]
	fitspath = @sprintf "data/DR16Q_Superset_v3/%04d/spec-%04d-%05d-%04d.fits" id.plate id.plate id.mjd id.fiberid

	hdulist = FITS(fitspath)
	flux = read(hdulist[2], "flux")
	loglam = read(hdulist[2], "loglam")
	close(hdulist)

	y = Vector{Float64}(flux)
	X = polynomial_features(Vector{Float64}(loglam), degree)
	n_best, a_best = dls_fit(X, y, k, removal_parameter)
	continuum = polynomial_features(Vector{Float64}(loglam), degree) * a_best
	normalised_flux = flux - continuum

	idx = (LOGLAMMIN .<= loglam) .& (loglam .<= LOGLAMMAX)
	flux_dset[i, :] = normalised_flux[idx]
end

fid_write = h5open("data/dr16q_superset.hdf5", "r+")
write_dataset(fid_write, "flux", flux_dset)
close(fid_write)
