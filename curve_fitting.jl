### A Pluto.jl notebook ###
# v0.12.17

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

# ╔═╡ 878e2a9c-2fe3-11eb-2200-1b16737bc3e3
using LinearAlgebra, Plots, PlutoUI

# ╔═╡ 90b0b1e0-3e06-11eb-048e-03da396a7956
md"# Polynomial Curve Fitting

BISHOP, Christopher M. *Pattern Recognition and Machine Learning*. New York: Springer, 2006. ISBN 978-0-387-31073-2.


We observe a real-valued input variable $x$ and we wish to use this observation to predict the value of a real-valued target variable $t$"

# ╔═╡ 16cdecc2-3e07-11eb-305c-e1c0bb77ff04
md"Data is generated from the function $\sin(2 \pi x)$ with random noise in the target values."

# ╔═╡ eac7be42-2fe4-11eb-049a-ed9f30a80313
function synthetic_data(n)
	# input values are generated uniformly in range (0, 1)
	x = rand(Float64, n)
	# target values are sin(2πx)
	# and adding random noise with a Gaussian distribution habing std 0.3
	t = sin.(2 * π .* x) + randn(Float64, n) * 0.3
	x, t
end

# ╔═╡ 8c6b7b78-3e07-11eb-333e-e387898b3b25
md"We are given a training set of $N$ observations of $x$. written $\mathsf{x} = (x_1, \dots, x_N)^\mathrm{T}$, together with corresponding observations of the values of $t$, denoted $\mathsf{t} = (t_1, \dots, t_N)^\mathrm{T}$."

# ╔═╡ 6e28d3b8-2fe3-11eb-0b03-b98c5ae53c5c
N = 10

# ╔═╡ 2eb33b02-2fe2-11eb-2d13-0d7d3921bd6e
x, t = synthetic_data(N);

# ╔═╡ 0bb4df7e-2fe3-11eb-0f41-3dbcf30f7707
begin
	x_range = range(0, 1, length=256)
	plot(x_range, sin.(2 * π .* x_range), label="sin(2πx)")
	scatter!(x, t, label="training set", xlabel="x", ylabel="t")
end

# ╔═╡ 82a85e70-3e08-11eb-36dd-059fbaf6b673
md"Our goal is to exploit this training set in order to make predictions of the value $\hat{t}$ of the target variable for some new value $\hat{x}$ of the input variable.

We shall fit the data using a polynomial function of the form

$y(x, \mathbf{w}) = \sum^M_{j = 0} w_j x^j$

where $M$ is the *order* of the polynomial.

Fitting the polynomial to the training data can be done by minimising an *error function*. Error function is geven by the sum of the squares of the errors between the predictions $y(x_n, \mathbf{w})$ for each data point $x_n$ and the corresponding target values $t_n$, so that we minimise

$E(\mathbf{w}) = \frac{1}{2} \sum_{n = 1}^N \{y(x_n, \mathbf{w}) - t_n\}^2$

where the factor of 1/2 is included for alter convenience."

# ╔═╡ 1bbd8c54-3e0a-11eb-1d9f-c37f71768611
function ols(X, t)
	(X' * X) ^ -1 * X' * t
end

# ╔═╡ 7c2478d0-3e0c-11eb-1b3c-836c1729a65f
function polynomial_features(x, M)
	n = length(x)
	X = Matrix{Float64}(undef, n, M + 1)
	for col = 1:M + 1
		X[:, col] = x .^ (col - 1)
	end
	return X
end

# ╔═╡ f3d9271e-3e0d-11eb-1d07-73bc4bf816b1
M_slider = @bind M Slider(0:9, default=3, show_value=true)

# ╔═╡ dca6d194-3e09-11eb-2186-59f8b8a09623
begin
	w_star = ols(polynomial_features(x, M), t)
	plot(x_range, sin.(2 * π .* x_range), label="sin(2πx)")
	scatter!(x, t, label="training data", xlabel="x", ylabel="t")
	plot!(
		x_range, polynomial_features(x_range, M) * w_star,
		label="M = $(M)", ylim=(-1.5, 1.5))
end

# ╔═╡ 3ae85302-3e0f-11eb-09c7-05679d09e5d7
md"We can obtain some quantitative insight into the dependence of the generalization performance on $M$ by considering a separate test set comprising 100 data points. For each choice of $M$, we can then evaluate the root-mean-square (RMS) error defined by

$E_\mathrm{RMS} = \sqrt{2 E(\mathbf{w^\star}) / N}$

in which the division by $N$ allows us to compare different sizes of data sets, and the square root ensures that $E_\mathrm{RMS}$ is measured on the same scale as the target variable $t$."

# ╔═╡ f8cc633e-3e0e-11eb-1117-2f02b6ad85b2
x_test, t_test = synthetic_data(100);

# ╔═╡ 673f2daa-3e10-11eb-3a9c-ef901b423b76
function rms(t̂, t)
	sqrt(sum((t̂ - t) .^ 2) / length(t))
end

# ╔═╡ 395cc8fc-3e10-11eb-3425-0b599a027c86
begin
	M_max = 9
	rms_train = zeros(M_max + 1)
	rms_test = zeros(M_max + 1)
	for M = 0:M_max
		X = polynomial_features(x, M)
		X_test = polynomial_features(x_test, M)
		w_star = ols(X, t)
		rms_train[M + 1] = rms(X * w_star, t)
		rms_test[M + 1] = rms(X_test * w_star, t_test)
	end
	scatter(0:M_max, rms_train, label="training", xlabel="M", ylabel="RMS")
	scatter!(0:M_max, rms_test, label="test")
end

# ╔═╡ 88aba1e2-3e12-11eb-32fe-51d73b65a8c7
md"It is also interesting to examine the behaviour of a given model as the size of the data ser is varied."

# ╔═╡ c7430594-3e12-11eb-3633-e1e8054da4d2
begin
	N_max = 100
	x_varied, t_varied = synthetic_data(N_max)
	@bind N_varied Slider(2:N_max, default=15, show_value=true)
end

# ╔═╡ e1a7fd54-3e12-11eb-2d15-e763f400d6af
begin
	w_varied = ols(polynomial_features(x_varied[1:N_varied], 9), t_varied[1:N_varied])
	plot(x_range, sin.(2 * π .* x_range), label="sin(2πx)")
	scatter!(
		x_varied[1:N_varied], t_varied[1:N_varied],
		label="training data", xlabel="x", ylabel="t")
	plot!(
		x_range, polynomial_features(x_range, 9) * w_varied,
		label="N = $(N_varied)", ylim=(-1.5, 1.5))
end

# ╔═╡ 05745fcc-3ea7-11eb-1e12-dbd34cfd652d
md"## Bayesian probabilities

We capture our assumptions about $\mathbf{w}$ in the form of a prior probability distribution $p(\mathbf{w})$. The effect of the observed data $\mathcal{D} = \{t_1, \dots t_N\}$ is expressed throught the conditional probability $p(\mathbf{w} | \mathcal{D})$. Bayes' theorem, which taks the form

$p(\mathbf{w} | \mathcal{D}) = \frac{p(\mathcal{D} | \mathbf{w}) p(\mathbf{w})}{p(\mathcal{D})}$

then allows us to evaluate the uncertainty in $\mathbf{w}$ *after* we have observer $\mathcal{D}$ in the form of the posterior probability $p(\mathbf{w} | \mathcal{D})$.

The quantity $p(\mathcal{D} | \mathbf{w})$ can be viewed as a function of $\mathbf{w}$, in which case it is called the *likelihood function*.

We can state Bayes' theorem in words

$\mathrm{posterior} \propto \mathrm{likelihood} \times \mathrm{prior}$

where all of these quantities are viewed as functions of $\mathbf{w}$.

Reducing the dependence on the prior is one motivation for so-called *noninformative* priors. However, Bayesian methods based on poor choices of prior can give poor results with high confidence."

# ╔═╡ d63e8ad0-3eaf-11eb-2a0c-b5bb8eb9cdd1
md"## The Gaussian Distribution

For the case of a single real-valued variable $x$, the the *normal* or *Gaussian* distribution is defined by

$\mathcal{N}(x | \mu, \sigma^2) = \frac{1}{(2 \pi \sigma^2)^{1/2}} \exp\left\{-\frac{1}{2 \sigma^2} (x - \mu) ^ 2\right\}$

which is governed by two parameters: $\mu$, called *mean*, and $\sigma^2$, called th *variance*. The square root of the variance, given by $\sigma$, is called *standard deviation*, and the recipprocal of the variance, written $\beta = 1 / \sigma^2$, is called the *precision*.

The average value of $x$ is given by

$\mathbb{E}[x] = \int^\infty_{-\infty} \mathcal{N}(x | \mu, \sigma^2) x \mathrm{d}x = \mu.$

Second order moment

$\mathbb{E}[x^2] = \int^\infty_{-\infty} \mathcal{N}(x | \mu, \sigma^2) x^2 \mathrm{d}x = \mu^2 + \sigma^2.$

It follows that the variance of $x$ is given by

$\mathrm{var}[x] = \mathbb{E}[x^2] - \mathbb{E}[x]^2 = \sigma^2$.

We are also interested in the Gaussian distribution defined over a $D$-dimensional vector $\mathbf{x}$ of continuous variables, which is given by

$\mathcal{N}(\mathbf{x} | \mathbf{\mu}, \mathbf{\Sigma}) = \frac{1}{(2 \pi)^{D / 2}} \frac{1}{|\mathbf{\Sigma}|^{1/2}} \exp\left\{ -\frac{1}{2} (\mathbf{x} - \mathbf{\mu})^\mathrm{T} \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{\mu}) \right\}.$"

# ╔═╡ 579f9176-3eb2-11eb-27fc-93ea43bb4a9e
md"## Curve Fitting Re-Visited

We can express our uncertainty over the value of the target variable using a probability distribution. We shall assume that, given the value of $x$, the corresponding value of $t$ has a Gaussian distribution with a mean equal to the value of $y(x, \mathbf{w})$. Thus we have

$p(t | x, \mathbf{w}, \beta) = \mathcal{N}(t | y(x, \mathbf{w}), \beta^{-1}).$

We now user the traing data $\{\mathsf{x}, \mathsf{t}\}$ to determine the values of the unknown parameters $\mathbf{w}$ and $\beta$ by maximum likelihood. If the data are assumed to be drawn idependently from the previous distribution, then the likelihood function is given by

$p(\mathsf{t} | \mathsf{x}, \mathbf{w}, \beta) = \prod^N_{n = 1} \mathcal{N}(t_n | y(x_n, \mathbf{w}), \beta^{-1}).$

It is convenient to maximize the logarithm of the likelihood function in the form

$\ln p(\mathsf{t} | \mathsf{x}, \mathbf{w}, \beta) = -\frac{\beta}{2} \sum^N_{n = 1} \{y(x_n, \mathbf{w}) - t_n\}^2 + \frac{N}{2} \ln \beta - \frac{N}{2} \ln(2 \pi).$

Lets us take a step towards a more Bayesian approach and introduce a prior distribution over the polynomial coefficients $\mathbf{w}$. For simplicity, let us consider a Gaussian distribution of the form

$p(\mathbf{w} | \alpha) = \mathcal{N}(\mathbf{w} | \mathbf{0}, \alpha^{-1} \mathbf{I}) = \left( \frac{\alpha}{2 \pi} \right)^{(M + 1) / 2} \exp \left\{ -\frac{\alpha}{2} \mathbf{w}^\mathrm{T} \mathbf{w} \right\}$

where $\alpha$ is the precision of the distribution, and $M + 1$ is the total number of elements in the vector $\mathbf{w}$. Using Bayes' theorem, the posterior distribution for $\mathbf{w}$ is proportional to the product of the prior distribution and the likelihood function

$p(\mathbf{w} | \mathsf{x}, \mathsf{t}, \alpha, \beta) \propto p(\mathsf{t} | \mathsf{x}, \mathbf{w}, \beta) p(\mathbf{w} | \alpha).$

We can now determine $\mathbf{w}$ by finding the most probable value of $\mathbf{w}$ given the data, in other words by maximizing the posterior distribution. This technique is called *maximum posterior*, or simpy *MAP*. We find that the maximum of the posterior is given by minimum of

$\frac{\beta}{2} \sum^N_{n = 1} \{y(x_n, \mathbf{w}) - t_n\}^2 + \frac{\alpha}{2} \mathbf{w}^\mathrm{t} \mathbf{w}.$"

# ╔═╡ 03b386e2-3eb9-11eb-0dce-85d1a7b84fc2
md"## Bayesian Curve Fitting

In a fully Bayesian approach, we should consistently apply the sum and product rules of probability, which requires, as we shall see shortly, that we integrate over all values of $\mathbf{w}$.

We wish to evaluate the predictive distribution $p(t | x, \mathsf{x}, \mathsf{t})$. Here we shall assume that the parameters $\alpha$ and $\beta$ are fixed and known in advance.

The predictive distribution is written in the form

$p(t | x, \mathsf{x}, \mathsf{t}) = \int p(t | x, \mathbf{w}) p(\mathbf{w} | \mathsf{x}, \mathsf{t}) \mathrm{d} \mathbf{w}.$

We shall see that tish posterior distribution is a Gaussian and can be evaluated analytically. The predictive distribution is given by a Gaussian of the form

$p(t | x, \mathsf{x}, \mathsf{t}) = \mathcal{N}(t | m(x), s^2(x))$

where the mean and variance are given by

$m(x) = \beta \mathbf{\phi}(x)^\mathrm{T} \mathbf{S} \sum^N_{n = 1} \mathbf{\phi}(x_n) t_n,$

$s^2(x) = \beta^{-1} + \mathbf{\phi}(x)^\mathrm{T} \mathbf{S} \mathbf{\phi}(x).$

Here the matrix $\mathbf{S}$ is given by

$\mathbf{S}^{-1} = \alpha \mathbf{I} + \beta \sum^N_{n = 1} \mathbf{\phi}(x_n) \mathbf{\phi}(x_n)^\mathrm{T}$

where $\mathbf{I}$ is th unit matrix, and we have defined the vector $\mathbf{\phi}(x)$ with elements $\phi_i(x_n) = x^i$ for $i = 0, \dots, M$."

# ╔═╡ 0dcdfaec-3ebc-11eb-3e53-c1ccea916de9
begin
	M_bayes = 9
	α = 5e-3
	β = (0.3 ^ -1) ^ 2    # the known noise variance
	ϕ = x -> polynomial_features([x], M_bayes)'
end;

# ╔═╡ e505caf2-3f76-11eb-1740-2dbeba8767e5
function compute_S(x_train, α, β, ϕ)
	inv(α * I + β * sum(ϕ(x_train[i]) * ϕ(x_train[i])' for i = 1:length(x_train)))
end

# ╔═╡ 9826d6f0-3ebc-11eb-254c-c1160590d8c0
function m(x, x_train, t_train, α, β, ϕ)
	S = compute_S(x_train, α, β, ϕ)
	pred = β * ϕ(x)' * S * sum(ϕ(x_train[i]) * t_train[i] for i = 1:length(x_train))
	return pred[1]
end

# ╔═╡ cc7bd846-3f76-11eb-18d4-c58448ac138a
function s²(x, x_train, α, β, ϕ)
	S = compute_S(x_train, α, β, ϕ)
	pred = β ^ -1 .+ ϕ(x)' * S * ϕ(x)
	return pred[1]
end

# ╔═╡ c62f363e-3f7d-11eb-26ba-a76c0a861ca2
@bind N_bayes Slider(2:N_max, default=15, show_value=true)

# ╔═╡ 6f3d1eac-3ebc-11eb-0e00-1f0c2b2ad500
begin
	plot(x_range, sin.(2 * π .* x_range), label="sin(2πx)")
	scatter!(
		x_varied[1:N_bayes], t_varied[1:N_bayes],
		label="training data", xlabel="x", ylabel="t")
	plot!(
		x_range,
		[m(x_range[i], x_varied[1:N_bayes], t_varied[1:N_bayes], α, β, ϕ) for i = 1:length(x_range)],
		ribbon=[sqrt.(s²(x_range[i], x_varied[1:N_bayes], α, β, ϕ) for i = 1:length(x_range))],
		ylim=(-1.5, 1.5))
end

# ╔═╡ Cell order:
# ╟─90b0b1e0-3e06-11eb-048e-03da396a7956
# ╠═878e2a9c-2fe3-11eb-2200-1b16737bc3e3
# ╟─16cdecc2-3e07-11eb-305c-e1c0bb77ff04
# ╠═eac7be42-2fe4-11eb-049a-ed9f30a80313
# ╟─8c6b7b78-3e07-11eb-333e-e387898b3b25
# ╠═6e28d3b8-2fe3-11eb-0b03-b98c5ae53c5c
# ╠═2eb33b02-2fe2-11eb-2d13-0d7d3921bd6e
# ╠═0bb4df7e-2fe3-11eb-0f41-3dbcf30f7707
# ╟─82a85e70-3e08-11eb-36dd-059fbaf6b673
# ╠═1bbd8c54-3e0a-11eb-1d9f-c37f71768611
# ╠═7c2478d0-3e0c-11eb-1b3c-836c1729a65f
# ╠═f3d9271e-3e0d-11eb-1d07-73bc4bf816b1
# ╠═dca6d194-3e09-11eb-2186-59f8b8a09623
# ╟─3ae85302-3e0f-11eb-09c7-05679d09e5d7
# ╠═f8cc633e-3e0e-11eb-1117-2f02b6ad85b2
# ╠═673f2daa-3e10-11eb-3a9c-ef901b423b76
# ╠═395cc8fc-3e10-11eb-3425-0b599a027c86
# ╟─88aba1e2-3e12-11eb-32fe-51d73b65a8c7
# ╠═c7430594-3e12-11eb-3633-e1e8054da4d2
# ╠═e1a7fd54-3e12-11eb-2d15-e763f400d6af
# ╟─05745fcc-3ea7-11eb-1e12-dbd34cfd652d
# ╟─d63e8ad0-3eaf-11eb-2a0c-b5bb8eb9cdd1
# ╟─579f9176-3eb2-11eb-27fc-93ea43bb4a9e
# ╟─03b386e2-3eb9-11eb-0dce-85d1a7b84fc2
# ╠═0dcdfaec-3ebc-11eb-3e53-c1ccea916de9
# ╠═e505caf2-3f76-11eb-1740-2dbeba8767e5
# ╠═9826d6f0-3ebc-11eb-254c-c1160590d8c0
# ╠═cc7bd846-3f76-11eb-18d4-c58448ac138a
# ╠═c62f363e-3f7d-11eb-26ba-a76c0a861ca2
# ╠═6f3d1eac-3ebc-11eb-0e00-1f0c2b2ad500
