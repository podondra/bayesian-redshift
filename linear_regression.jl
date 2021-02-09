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

# ╔═╡ 58951756-2d80-11eb-16e5-73b4b46539b1
begin
	using HDF5
	using LinearAlgebra
	using Plots
	include("Evaluation.jl"); import .Evaluation
	include("Utils.jl"); import .Utils
end

# ╔═╡ f7494ac2-410b-11eb-23f5-a5e328c708ed
using Random, Distributions, StatsPlots, PlutoUI

# ╔═╡ 477e848c-403f-11eb-1f94-11d67a92db4b
md"# Linear Models for Regression

BISHOP, Christopher M. *Pattern Recognition and Machine Learning*. New York: Springer, 2006. ISBN 978-0-387-31073-2.

Given a training data set comprising $N$ observations $\{\mathbf{x}_n\}$, where $n = 1, \dots, N$, together with corresponding target values $\{t_n\}$, the goal is to predict the value of $t$ for a new value of $\mathbf{x}$. From a probabilistic perspective, we aim to model the predictive distributoin $p(t | \mathbf{x})$ because this expresses our uncertainty about the value of $t$ for each value of $\mathbf{x}$."

# ╔═╡ 57350c8e-4047-11eb-3b1f-7f66e571cefc
md"## Linear Basis Function Models

The simplest linear model for regression (*linear regression*) is one that involves a linear combination of input variables

$y(\mathbf{x}, \mathbf{w}) = w_0 + w_1 x_1 + \dots + w_D x_D$

where $\mathbf{x} = (x_1, \dots, x_D) ^ \mathrm{T}$. We extend the class of models by considering linear combinations of fixed nonlinear functions of the input variables, of the form

$y(\mathbf{x}, \mathbf{w}) = w_0 + \sum_{j = 1}^{M - 1} w_j \phi_j(\mathbf{x})$

where $\phi_j(\mathbf{x})$ are known as *basis functions*.

The parameter $w_0$ allows for any fixed offset in the data and is sometimes called a *bias* parameter. It is often convenient to define an additional dummy 'basis function' $\phi_0(\mathbf{x}) = 1$ so that

$y(\mathbf{x}, \mathbf{w}) = \sum_{j = 0}^{M - 1} w_j \phi_j(\mathbf{x}) = \mathbf{w} ^ \mathrm{T} \boldsymbol{\phi}(\mathbf{x})$

where $\mathbf{w} = (w_0, \dots, w_{M - 1}) ^ \mathrm{T}$ and $\boldsymbol{\phi} = (\phi_0, \dots, \phi_{M - 1}) ^ \mathrm{T}$.

Indeed, much of our discussion will be equally applicable to the situation in which the vector $\phi(\mathbf{x})$ of basis functions is simply the identity $\boldsymbol{\phi}(\mathbf{x}) = \mathbf{x}$."

# ╔═╡ 56bda126-4047-11eb-01e5-1b21bfafe659
md"### Maximum Likelihood and Least Squares

We assume that the target variable $t$ is given by a deterministic function $y(\mathbf{x}, \mathbf{w})$ with additive Gaussian noise so that

$t = y(\mathbf{x}, \mathbf{w}) + \epsilon$

where $\epsilon$ is a zero mean Gaussian random variable with precision (inverse variance) $\beta$. Thus we can write

$p(t | \mathbf{x}, \mathbf{w}, \beta) = \mathcal{N}(t | y(\mathbf{x}, \mathbf{w}), \beta^{-1}).$

Recall that, if we assume a squared loss function, then the optimal prediction, for a new value of $\mathbf{x}$, will be given by the conditional mean of the target variable. In the case of a Gaussian confition distribution, the conditional mean will be

$\mathbb{E}[t | \mathbf{x}] = \int tp(t | \mathbf{x}) \mathrm{d}t = y(\mathbf{x}, \mathbf{w}).$

Now consider a data set of inputs $\mathbf{X} = \{\mathbf{x}_1, \dots, \mathbf{x}_N\}$ with corresponding target values $t_1, \dots, t_N$. We group the target variables $\{t_n\}$ into a column vector that we denote by $\mathsf{t}$. Making the assumption that these data points are drawn independently, we obtain the following expression for the likelihood function, which is a function of these adjustable parameters $\mathbf{w}$ and $\beta$, in the form

$p(\mathsf{t} | \mathbf{X}, \mathbf{w}, \beta) = \prod \mathcal{N}(t_n | \mathbf{w}^\mathrm{T} \boldsymbol{\phi}(\mathbf{x}_n), \beta^{-1}).$"

# ╔═╡ 5ded6f22-2d81-11eb-1ecf-15831fa58960
begin
	fid = h5open("data/dr16q_superset.hdf5", mode="r")
	X = read(fid["X_tr"])'
	t = convert.(Float32, read(fid["z_tr"]))
	X_va = read(fid["X_va"])'
	t_va = convert.(Float32, read(fid["z_va"]))
	N = size(X, 1)
	N_va = size(X_va, 1)
	close(fid)
	N, size(X), size(t)
end

# ╔═╡ 18ca93fc-4da7-11eb-03ad-95d833f5ffbc
N_va, size(X_va), size(t_va)

# ╔═╡ a6f552ac-405c-11eb-3172-cd1d78946585
md"Note that $\mathbf{x}$ will always appear in the set of conditioning variables, and so from now on we will drop the explicit $\mathbf{x}$ from expressions. Taking the logarithm of the likelihood function, and making use of the standard form for the univariate Gaussian, we have

$\ln p(\mathsf{t} | \mathbf{w}, \beta) = \frac{N}{2} \ln \beta - \frac{N}{2} \ln (2 \pi) - \beta E_D(\mathbf{w})$

where the sum-of-squares error function is defined by

$E_D(\mathbf{w}) = \frac{1}{2} \sum_{n = 1}^N \{t_n - \mathbf{w}^\mathrm{T}\boldsymbol{\phi}(\mathbf{x}_n)\}^2.$

Having written the likelihood function, we can use maximum likelihood to determine $\mathbf{w}$ and $\beta$. Consider first the maximization with respect to $\mathbf{w}$. The gradient of the log likelihood functoin takes the form

$\nabla \ln p(\mathsf{t} | \mathbf{w}, \beta) = \sum^N_{n = 1} \{t_n - \mathbf{w}^\mathrm{T} \boldsymbol{\phi}(\mathbf{x}_n)\} \boldsymbol{\phi}(\mathbf{x}_n)^\mathrm{T}$

Setting this gradient to zero gives

$0 = \sum^N_{n = 1} t_n \boldsymbol{\phi}(\mathbf{x}_n)^\mathrm{T} - \mathbf{w}^\mathrm{T} \left( \sum^N_{n = 1} \boldsymbol{\phi}(\mathbf{x}_n) \boldsymbol{\phi}(\mathbf{x}_n)^\mathrm{T} \right).$

Solving for $\mathbf{w}$ we obtain

$\mathbf{w}_\mathrm{ML} = (\boldsymbol\Phi^\mathrm{T} \boldsymbol\Phi)^{-1} \boldsymbol\Phi^\mathrm{T}\mathsf{t}$

which are known as the *normal equations* for the least squares problem. Here $\boldsymbol\Phi$ is an $N \times M$ matrix, called the *design matrix*, whose elemnt are given by $\Phi_{nj} = \phi_j(\mathbf{x}_n)$."

# ╔═╡ 80db7738-405d-11eb-04fb-29b8aa40570a
# add bias basis function
#Φ = X
Φ = [ones(Float32, N) X]

# ╔═╡ e925676e-2d8a-11eb-114e-9f5a397e579d
w_ml = inv(Φ' * Φ) * Φ' * t

# ╔═╡ 7fa302e4-405f-11eb-3006-d3b6ce5fbc6f
y = Φ * w_ml

# ╔═╡ 70a584aa-405d-11eb-0f58-f7c68ee0f7d4
md"We can also maximize the log likelihood function with respect to the noise precision parameter $\beta$, giving

$\frac{1}{\beta_{\mathrm{ML}}} = \frac{1}{N} \sum_{n = 1}^N \{t_n - \mathbf{w}^\mathrm{T}_\mathrm{ML} \boldsymbol\phi(\mathbf{x}_n)\}^2$

and so we see that the inverse of the noise precision is given by the residual variance of the target values around the regression function."

# ╔═╡ 2feba2f2-405e-11eb-3b49-7bf26d78ddb9
σ²_ml = sum((t[n] - w_ml' * Φ[n, :]) ^ 2 for n = 1:N) / N

# ╔═╡ 0e6e5b3c-405f-11eb-31a0-15f78cb09cfc
σ_ml = sqrt(σ²_ml)

# ╔═╡ 44665762-4da7-11eb-3142-d5b022bfbb32
Φ_va = [ones(Float32, N_va) X_va]

# ╔═╡ 5662d814-4da7-11eb-2796-21b20d8cec23
y_va = Φ_va * w_ml

# ╔═╡ e6ddadda-6abf-11eb-3713-c3b77ff03a1d
begin
	i = rand(1:N_va)
	Utils.plot_spectrum(X_va[i, :])
	Utils.plot_spectral_lines!(t_va[i])
	Utils.plot_spectral_lines!(y_va[i], color=:red, location=:bottom)
end

# ╔═╡ e5aa288c-4da3-11eb-0d6a-17ccbbc907ee
Evaluation.rmse(t_va, y_va)

# ╔═╡ 0b89afc8-4da4-11eb-14e6-4d62de20f275
md"$\Delta v = c \frac{|z - z_\mathrm{VI}|}{1 + z_\mathrm{VI}}$"

# ╔═╡ 2af527c4-4da5-11eb-114a-7fb9772594b3
Evaluation.catastrophic_redshift_ratio(t_va, y_va)

# ╔═╡ c1424a2e-4060-11eb-25f0-91259901fbfb
md"## Bayesian Linear Regression

We turn to Bayesian treatment of linear regression, which will avoid the over-fitting problem of maximum likelihood, and which will also lead to automatic methods of determining model complexity using the training data alone."

# ╔═╡ c2c5fc70-4104-11eb-2f83-d11f8ecd277f
md"### Parameter Distribution

We begin by introducing a prior probability distribution over the model parameters $\mathbf{w}$. For the moment, we shall treat the noise precision parameter $\beta$ as a known constant. First note that the likelihood function $p(\mathsf{t} | \mathbf{w})$ is the exponential of a quadratic function of $\mathbf{w}$. The corresponding conjugate prior is therefore given by a Gaussian distribution of the form

$p(\mathbf{w}) = \mathcal{N}(\mathbf{w} | \mathbf{m}_0, \mathbf{S}_0)$

having mean $\mathbf{m}_0$ and covariance $\mathbf{S}_0$.

Next we compute the posterior distribution, which is proportional to the product of the likelihood function and the prior. Due to the choice of a conjugate Gaussian prior distribution, the posterior will also be Gaussian. The posterior distribution is in the form

$p(\mathbf{w} | \mathsf{t}) = \mathcal{N}(\mathbf{w} | \mathbf{m}_N, \mathbf{S}_N)$

where

$\mathbf{m}_N = \mathbf{S}_N (\mathbf{S}^{-1}_0 \mathbf{m}_0 + \beta \boldsymbol\Phi^\mathrm{T} \mathsf{t})$

$\mathbf{S}^{-1}_N = \mathbf{S}^{-1}_0 + \beta \boldsymbol\Phi^\mathrm{T} \boldsymbol\Phi.$

Note that because the posterior distribution is Gaussian, its mode coincides with its mean. Thus the maximum posterior weight vector is simply given by $\mathbf{w}_\mathrm{MAP} = \mathbf{m}_N$. If we consider an infinitely broad prior $\mathbf{S}_0 = \alpha^{-1} \mathbf{I}$ with $\alpha \to 0$, the mean $\mathbf{m}_N$ of the posterior distribution reduces to the maximum likelihood value $\mathbf{w}_\mathrm{ML}$. Similarly, if $N = 0$, then the posterior distribution reverts to the prior.

For the remainder, we shall consider a particular form of Gaussian prior in order to simplify the treatment. Specifically, we consider a zero-mean isotropic Gaussian governed by a single precision parameter $\alpha$ so that

$p(\mathbf{w}) = \mathcal{N}(\mathbf{w} | \mathbf{0}, \alpha^{-1} \mathbf{I})$

and the corresponding posterior distribution over $\mathbf{w}$ is then given with

$\mathbf{m}_N = \beta \mathbf{S}_N \boldsymbol\Phi^\mathrm{T} \mathsf{t}$

$\mathbf{S}^{-1}_N = \alpha \mathbf{I} + \beta \boldsymbol\Phi^\mathrm{T} \boldsymbol\Phi.$"

# ╔═╡ 20627ada-410b-11eb-3733-8b1b8ca0bcaf
function synthetic_data(N)
	# input values are generated uniformly in range (-1, 1)
	d = Uniform(-1, 1)
	x = rand(d, N)
	# target values are f(x, a) = a₀ + a₁ x
	a₀ = -0.3
	a₁ = 0.5
	f = x -> a₀ + a₁ * x
	# and adding random noise with a Gaussian distribution with std 0.2
	d = Normal(0, 0.2)
	t = f.(x) + rand(d, N)
	x, t
end

# ╔═╡ bc762fd6-410c-11eb-13c8-9fb7f2478ac4
begin
	N_syn = 20
	x_syn, t_syn = synthetic_data(N_syn)
	scatter(x_syn, t_syn, xlabel="x", ylabel="t", legend=:none)
end

# ╔═╡ 527da3e0-4114-11eb-3af1-5ff21c7040e6
begin
	β_syn = (1 / 0.2) ^ 2
	α_syn = 2.0
	S0_syn = α_syn ^ -1 * I
	m0_syn = zeros(2)
	prior = MvNormal(m0_syn, S0_syn)
	
	N_range = 256
	w0_range = range(-1, 1, length=N_range)
	w1_range = range(-1, 1, length=N_range)
	prior_density = zeros(N_range, N_range)
	for i = 1:N_range, j = 1:N_range
		prior_density[i, j] = pdf(prior, [w0_range[i], w1_range[j]])
	end
	
	contour(w0_range, w1_range, prior_density', xlabel="w₀", ylabel="w₁")
	title!("prior")
	scatter!([-0.3], [0.5], legend=:none)
end

# ╔═╡ 98f8c482-4121-11eb-131d-2157366cfbfb
md"### Predictive Distribution

In practice, we are not usually interested in the value of $\mathbf{w}$ itself but rather in making predictions of $t$ for new values of $\mathbf{x}$. This requires that we evaluate the *predictive distribution* defined by

$p(t | \mathsf{t}, \alpha, \beta) = \int p(t | \mathbf{w}, \beta) p(\mathbf{w} | \mathsf{t}, \alpha, \beta) \mathrm{d}\mathbf{w}$

in which $\mathsf{t}$ is the vector of target values from the training set, and we have omitted the corresponding input vectors from the right-hand side of the conditioning statements to simplify the notation. We see that it involves the convolution of two Gaussian distributions, and so we see that the predtictive distribution takes the form

$p(t | \mathbf{x}, \mathsf{t}, \alpha, \beta) = \mathcal{N}(t | \mathbf{m}_N^\mathrm{T} \boldsymbol\phi(\mathbf{x}), \sigma^2_N(\mathbf{x}))$

where the variance $\sigma^2_N(\mathbf{x})$ of the predictive distribution is given by

$\sigma^2_N(\mathbf{x}) = \frac{1}{\beta} + \boldsymbol\phi(\mathbf{x})^\mathrm{T} \mathbf{S}_N \boldsymbol\phi(\mathbf{x}).$

The first term represent the noise on the data whereas the second term reflects the uncertainty associated with the parameters $\mathbf{w}$."

# ╔═╡ 7411237a-4113-11eb-2758-19dedb770b85
@bind N_varied Slider(1:20, default=1, show_value=true)

# ╔═╡ 3ce55264-4112-11eb-3b8c-497d16731428
begin
	Φ_syn = [ones(N_syn) x_syn][1:N_varied, :]
	S_syn = inv(α_syn * I + β_syn * Φ_syn' * Φ_syn)
	m_syn = β_syn * S_syn * Φ_syn' * t_syn[1:N_varied]
	posterior = MvNormal(m_syn, S_syn)

	posterior_density = zeros(N_range, N_range)
	for i = 1:N_range, j = 1:N_range
		posterior_density[i, j] = pdf(posterior, [w0_range[i], w1_range[j]])
	end
	
	l = @layout [a b]
	
	p1 = contour(w0_range, w1_range, posterior_density', xlabel="w₀", ylabel="w₁")
	title!("posterior (N = $(N_varied))")
	scatter!([-0.3], [0.5], legend=:none)
	
	p2 = scatter(
		x_syn[1:N_varied], t_syn[1:N_varied], xlabel="x", ylabel="t", legend=:none,
		xlim=(-1, 1), ylim=(-1, 1))
	x_range = range(-1, 1, length=N_range)
	Φ_range = [ones(N_range) x_range]
	w = rand(posterior, 6)
	plot!(x_range, Φ_range * w, color=:green)
	plot!(
		x_range, Φ_range * m_syn, color=:red,
		ribbon=sqrt.([1 / β_syn + Φ_range[i, :]' * S_syn * Φ_range[i, :] for i = 1:length(x_range)]))

	plot(p1, p2, layout=l)
end

# ╔═╡ 8d710c68-4da7-11eb-0ff1-b51ba31b19ef
md"Note that, if both $\mathbf{w}$ and $\beta$ are treated as unknown, then we can introduce a conjugate prior distribution $p(\mathbf{w}, \beta)$ that will be given by a Gaussian-gamma distribution. In this case, the predictive distribution is a Student's t-distribution."

# ╔═╡ e1f04b28-4da7-11eb-179b-7bc8a5391136
begin
	# TODO introduce a conjugate prior ditribution p(w, β)
	β = 1.0
	α = 2.0
	S = inv(α * I + β * Φ' * Φ)
	m = β * S * Φ' * t
end

# ╔═╡ f128c9ac-4da8-11eb-029b-45d1c61bb4b8
y_va_bayes = Φ_va * m

# ╔═╡ a1550f8e-4da9-11eb-27ef-f5d76d693e25
Evaluation.rmse(t_va, y_va_bayes)

# ╔═╡ 174a9e60-4da9-11eb-25ee-7f4d092a8e59
y_va_std = sqrt.([1 / β + Φ_va[i, :]' * S * Φ_va[i, :] for i = 1:N_va])

# ╔═╡ 5c3e572a-4da9-11eb-0270-2b848a6b2595
histogram(y_va_std, legend=:none)

# ╔═╡ d3e4554a-4da9-11eb-3975-b526831e612d
Evaluation.catastrophic_redshift_ratio(t_va, y_va_bayes)

# ╔═╡ be5f3988-4da9-11eb-06b2-5767b9f52edd
begin
	idx = y_va_std .< 1.001
	sum(idx), Evaluation.catastrophic_redshift_ratio(t_va[idx], y_va_bayes[idx])
end

# ╔═╡ Cell order:
# ╟─477e848c-403f-11eb-1f94-11d67a92db4b
# ╟─57350c8e-4047-11eb-3b1f-7f66e571cefc
# ╟─56bda126-4047-11eb-01e5-1b21bfafe659
# ╠═58951756-2d80-11eb-16e5-73b4b46539b1
# ╠═5ded6f22-2d81-11eb-1ecf-15831fa58960
# ╠═18ca93fc-4da7-11eb-03ad-95d833f5ffbc
# ╟─a6f552ac-405c-11eb-3172-cd1d78946585
# ╠═80db7738-405d-11eb-04fb-29b8aa40570a
# ╠═e925676e-2d8a-11eb-114e-9f5a397e579d
# ╠═7fa302e4-405f-11eb-3006-d3b6ce5fbc6f
# ╟─70a584aa-405d-11eb-0f58-f7c68ee0f7d4
# ╠═2feba2f2-405e-11eb-3b49-7bf26d78ddb9
# ╠═0e6e5b3c-405f-11eb-31a0-15f78cb09cfc
# ╠═44665762-4da7-11eb-3142-d5b022bfbb32
# ╠═5662d814-4da7-11eb-2796-21b20d8cec23
# ╠═e6ddadda-6abf-11eb-3713-c3b77ff03a1d
# ╠═e5aa288c-4da3-11eb-0d6a-17ccbbc907ee
# ╟─0b89afc8-4da4-11eb-14e6-4d62de20f275
# ╠═2af527c4-4da5-11eb-114a-7fb9772594b3
# ╟─c1424a2e-4060-11eb-25f0-91259901fbfb
# ╟─c2c5fc70-4104-11eb-2f83-d11f8ecd277f
# ╠═f7494ac2-410b-11eb-23f5-a5e328c708ed
# ╠═20627ada-410b-11eb-3733-8b1b8ca0bcaf
# ╠═bc762fd6-410c-11eb-13c8-9fb7f2478ac4
# ╠═527da3e0-4114-11eb-3af1-5ff21c7040e6
# ╟─98f8c482-4121-11eb-131d-2157366cfbfb
# ╠═7411237a-4113-11eb-2758-19dedb770b85
# ╠═3ce55264-4112-11eb-3b8c-497d16731428
# ╟─8d710c68-4da7-11eb-0ff1-b51ba31b19ef
# ╠═e1f04b28-4da7-11eb-179b-7bc8a5391136
# ╠═f128c9ac-4da8-11eb-029b-45d1c61bb4b8
# ╠═a1550f8e-4da9-11eb-27ef-f5d76d693e25
# ╠═174a9e60-4da9-11eb-25ee-7f4d092a8e59
# ╠═5c3e572a-4da9-11eb-0270-2b848a6b2595
# ╠═d3e4554a-4da9-11eb-3975-b526831e612d
# ╠═be5f3988-4da9-11eb-06b2-5767b9f52edd
