### A Pluto.jl notebook ###
# v0.12.17

using Markdown
using InteractiveUtils

# ╔═╡ 58951756-2d80-11eb-16e5-73b4b46539b1
using HDF5, LinearAlgebra, Plots

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
	fid = h5open("data/dataset.hdf5", mode="r")
	dset = fid["X_tr"]
	X = read(dset)'
	t = convert.(Float32, read(fid["z_tr"]))
	N = size(X, 1)
	close(fid)
	N, size(X), size(t)
end

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

# ╔═╡ 2edf19a4-2e32-11eb-1bcb-a1e631a4b25b
function z_to_wave(z)
	(1 + z) * 1909
end

# ╔═╡ bdceb186-2e2f-11eb-0225-d7891641ec07
begin
	wave = 10 .^ range(3.5836, 3.9559, length=512)
	plot(wave, X[1, :])
	vline!([z_to_wave(y[1])])
end

# ╔═╡ 5641e4c0-2e2f-11eb-0cda-41a92f0fbdb1
histogram(t - y)

# ╔═╡ Cell order:
# ╟─477e848c-403f-11eb-1f94-11d67a92db4b
# ╟─57350c8e-4047-11eb-3b1f-7f66e571cefc
# ╟─56bda126-4047-11eb-01e5-1b21bfafe659
# ╠═58951756-2d80-11eb-16e5-73b4b46539b1
# ╠═5ded6f22-2d81-11eb-1ecf-15831fa58960
# ╟─a6f552ac-405c-11eb-3172-cd1d78946585
# ╠═80db7738-405d-11eb-04fb-29b8aa40570a
# ╠═e925676e-2d8a-11eb-114e-9f5a397e579d
# ╠═7fa302e4-405f-11eb-3006-d3b6ce5fbc6f
# ╟─70a584aa-405d-11eb-0f58-f7c68ee0f7d4
# ╠═2feba2f2-405e-11eb-3b49-7bf26d78ddb9
# ╠═0e6e5b3c-405f-11eb-31a0-15f78cb09cfc
# ╠═2edf19a4-2e32-11eb-1bcb-a1e631a4b25b
# ╠═bdceb186-2e2f-11eb-0225-d7891641ec07
# ╠═5641e4c0-2e2f-11eb-0cda-41a92f0fbdb1
