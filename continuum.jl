### A Pluto.jl notebook ###
# v0.12.16

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 6ce82c08-3081-11eb-2d98-014715915a87
using FITSIO, LinearAlgebra, Plots, PlutoUI

# ╔═╡ 48e3fa0c-3613-11eb-0b02-b5ab3d5ddece
function polynomial_features(x::Vector{Float64}, degree::Integer)::Matrix{Float64}
	n = length(x)
	X = Matrix{Float64}(undef, n, degree + 1)
	for col = 1:degree + 1
		X[:, col] = x .^ (col - 1)
	end
	return X
end

# ╔═╡ 1eea25d8-34af-11eb-3771-07b752765bf2
function fit(X::Matrix{Float64}, y)::Vector{Float64}
    return inv(X' * X) * X' * y
end

# ╔═╡ eca7a9c0-35fe-11eb-0cfe-c7aa071231f7
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

# ╔═╡ 6531ac7e-3085-11eb-0633-a5bf7210fb85
filenames = [
	"data/sdss_dr14/7613/spec-7613-56988-0137.fits",
	"data/sdss_dr14/1504/spec-1504-52940-0474.fits",
	"data/sdss_dr14/7692/spec-7692-57064-0969.fits",
	"data/sdss_dr14/8375/spec-8375-57520-0291.fits"]

# ╔═╡ dd519002-35fd-11eb-06a5-6fd483757583
@bind filename Select(filenames)

# ╔═╡ ec85421c-308f-11eb-250b-5598eab19992
begin
	hdul = FITS(filename)
	flux = read(hdul[2], "flux")
	loglam = read(hdul[2], "loglam")
	plot(loglam, flux, label=filename)
end

# ╔═╡ a7583e50-3612-11eb-10ed-d9b759a61fab
begin
	k = 2
	removal_parameter = 0.9
	degree = 3
	y = Vector{Float64}(flux)
	X = polynomial_features(Vector{Float64}(loglam), degree)
	a = fit(X, y)

	scatter(loglam, flux, legend=:none)
	plot!(X[:, 2], X * a)
end

# ╔═╡ 5c3e3504-35ff-11eb-2986-23c618041f56
function dls_fit(X::Matrix{Float64}, y::Vector{Float64},
		k::Int64, removal_parameter::Float64)
	n, m = size(X)
	subset_n, subset_dls = Int64[], Float64[]

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

# ╔═╡ ec54b524-3621-11eb-385d-b51c22f49c9c
n_best, a_best = dls_fit(X, y, 2, 0.9)

# ╔═╡ 9250244a-35ff-11eb-0788-1f7af6a66e01
begin
	scatter(X[n_best + 1:end, 2], y[n_best + 1:end], label="distant points")
	scatter!(X[1:n_best, 2], y[1:n_best], label="close points")
	continuum = polynomial_features(Vector{Float64}(loglam), degree) * a_best
	p1 = plot!(loglam, continuum, label="continuum")
	p2 = plot(loglam, flux - continuum, label="normalised")
	plot(p1, p2, layout=(2, 1))
end

# ╔═╡ Cell order:
# ╠═6ce82c08-3081-11eb-2d98-014715915a87
# ╠═48e3fa0c-3613-11eb-0b02-b5ab3d5ddece
# ╠═1eea25d8-34af-11eb-3771-07b752765bf2
# ╠═eca7a9c0-35fe-11eb-0cfe-c7aa071231f7
# ╠═5c3e3504-35ff-11eb-2986-23c618041f56
# ╠═6531ac7e-3085-11eb-0633-a5bf7210fb85
# ╠═dd519002-35fd-11eb-06a5-6fd483757583
# ╠═ec85421c-308f-11eb-250b-5598eab19992
# ╠═a7583e50-3612-11eb-10ed-d9b759a61fab
# ╠═ec54b524-3621-11eb-385d-b51c22f49c9c
# ╠═9250244a-35ff-11eb-0788-1f7af6a66e01
