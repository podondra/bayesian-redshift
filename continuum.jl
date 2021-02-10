### A Pluto.jl notebook ###
# v0.12.20

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

# ╔═╡ 1be4745e-4dd9-11eb-0817-bd35c38a35f8
md"# Advanced Fit Technique for Astrophysical Spectra

BUKVIĆ, S., Dj. SPASOJEVIĆ a V. ŽIGMAN. Advanced fit technique for astrophysical spectra. *Astronomy & Astrophysics*. EDP Sciences, 2008, roč. 477, č. 3, s. 967–977. DOI: 10.1051/0004-6361:20065969


A robust method of data fitting convinient for dealing with astrophysical spectra contaminated by a large fraction of outliers or unwelcome features of the the spectrum are to be considered as formal outliers (spectral lines while estimating continuum).

The best-fit parameters are obtained by the least-square method on a subset having maximum value of DLS that is the largest subset free of outliers."

# ╔═╡ b1725bea-4ddd-11eb-3e57-4320a3072836
md"## Merit Function

Consider a dataset $s_0 = \{(x_i, y_i) : i = 1, \dots, n_0\}$ consisting of $n_0$ experimental points and a family of model functions $f(x; a_1, \dots, a_m)$ depending on $x$ and on the parameters $a_1, \dots, a_m \equiv \mathbf{a}$, specifying a model function from the family.

In order to identify the best subset $s_b$, we introduce the *density of least squares*: the quantity suitable for characterizing the intial data set $s_0$ and any particular subset $s$, in the following way

$D_k(s) = \frac{\sum_s d_i^2}{d_\max^k(s)}$

where *distance* $d_i = |y_i - f(x_i; \mathbf{a}(s))|$ is the absolute value of the deviation of $y_i$ from the model function $f(x_i; \mathbf{a}(s))$. The parameters $\mathbf{a}(s)$ are the best-fit parameters, obtained by applying the ordinary least square (OLS) method to the subset $s$, soe the the numeerator is the corresponding OLS sum for that subset. Analogously, $d_\max(s) = \max\{d_i\}$ is the maximal distance, which specifies the *width* of subset $s$, relative to the model function $f(x_i; \mathbf{a}(s))$. The exponent $2 \le k \lt 3$ is a real number whose value is yet to be determined (the choice of $k = 2$ further simplifies the method).

The *best subset* $s_b$ has the maximum density of the least squares. Its OLS best-fit parameters $\mathbf{a}(s_b)$ are considered to yield the most reliable estimate of the features of interest.

To conform to standard terminalogy, we introduce the merit function

$\mu_k(s) = -D_k(s)$

which is a function of subset $s$ of the dataset $s_0$. On subset $s_b$, the merit function $\mu_k(s)$ attains its minimum value whereas $D_k(s)$ attains its maximum value.

This approach is called the *density of least-square (DLS) method. The points from teh subset $s_b$ are *close points* while all other points are *distant points* or *outliers* with respect to the model function. In the same sense, the *width of the best subset* $d_b = d_\max(s_b)$ can be conveniently used to set a discrimination level between distant and close points."

# ╔═╡ 78d54e24-4de1-11eb-3d9f-5febd423dec9
md"If individual errors $\sigma_i$ are assigned to corresponding $y_i$ then (instead of geometrical distanced $d_i$) *relative distances* $\delta_i = d_i / \sigma_i$ should be used. Accordingly, the density of *weighted* least squares is defined by

$D_{\mathrm{W}, k}(s) = \frac{\sum_s \delta_i^2}{\delta_\max^k(s)}.$"

# ╔═╡ 6d3cdeae-4de3-11eb-38d5-411efae4dcdb
md"We define the ordered collection of data subsets by iteraing the following two-stage procedure. We first calculate the OLS best-fit model function for the given subset $s$ and then remove from $s$ all the points such that their distances $d_i \ge r d_\max(s)$ where $0 \lt r \le 1$ is a *removal* parameter. In second stage, we apply OLS only to the retained data points and recalculate distances $d_i$ of the remaining data points from the new best-fit model function. If there are points with distances larger than the starting $d_\max(s)$ we remove them as well. We repreate theses steps until all points with distances $d_i \ge d_\max(s)$ are removed. The smaller subset of retained points with new $d_\max$ is now ready for the next iteration cycle. Points eliminated from $s$ by this procedure constitute a *layer of outer points* for subset $s$ (simply *layer*).

By using the foregoing procedure, we iteratively remove layer-by-layer form the initial dataset $s_0$ and we eventually obtain an ordered collection of dats subsets $\mathcal{C} = \{s_0 \supset \dots \supset s_p\}$ where $p$ is the number of subsets in the collection. After each iteration, we calculate $D_k(s)$. We terminate the build-up of collection $\mathcal{C}$ whenever $n \lt m + 3$ (subset of $n$ data points and $m$-parameter model function). As the best subset $s_b$, we accept the subset from collection $\mathcal{C}$ having a maximum value of DLS."

# ╔═╡ 48e3fa0c-3613-11eb-0b02-b5ab3d5ddece
function polynomial_features(x::Vector{Float64}, degree::Int64)::Matrix{Float64}
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
filenames = "data/DR16Q_Superset_v3/" .* [
	"7613/spec-7613-56988-0137.fits",
	"7692/spec-7692-57064-0969.fits",
	"8375/spec-8375-57520-0291.fits"]

# ╔═╡ dd519002-35fd-11eb-06a5-6fd483757583
@bind filename Select(filenames)

# ╔═╡ ec85421c-308f-11eb-250b-5598eab19992
begin
	hdul = FITS(filename)
	flux = read(hdul[2], "flux")
	loglam = read(hdul[2], "loglam")
	plot(loglam, flux, label=filename, ylabel="flux", xlabel="wavelength")
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
function dls_fit(
		X::Matrix{Float64}, y::Vector{Float64},
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
	scatter!(X[1:n_best, 2], y[1:n_best], label="close points", legend=:topleft)
	continuum = polynomial_features(Vector{Float64}(loglam), degree) * a_best
	p1 = plot!(loglam, continuum, label="continuum")
	p2 = plot(loglam, flux - continuum, label="normalised")
	plot(p1, p2, layout=(2, 1))
end

# ╔═╡ Cell order:
# ╟─1be4745e-4dd9-11eb-0817-bd35c38a35f8
# ╟─b1725bea-4ddd-11eb-3e57-4320a3072836
# ╟─78d54e24-4de1-11eb-3d9f-5febd423dec9
# ╟─6d3cdeae-4de3-11eb-38d5-411efae4dcdb
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
