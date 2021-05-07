### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# ╔═╡ 5c7adecc-aefa-11eb-2bb5-f5778d7edcb2
md"# Prediction of Spectroscopic Redshift with a Bayesian Convolutional Network

For IMRaD reference: [Writing a Scientific Research Report (IMRaD)](https://writingcenter.gmu.edu/guides/writing-an-imrad-report).
"

# TODO abstract

# ╔═╡ 1a436fc9-d74a-441c-8125-1af54d17fe97
md"## Introduction

Why this research is important or necessary or important?
1. Motivation is to reduce the number of visual classification in catalogues of QSOs.
1. We compare predictoin of redshift as regression and classification.
1. We experiment with *real* SDSS data (DR12Q and DR16Q).
1. We are better than QuasarNet? (TODO: verify in DR12Q)
1. We have correct model confidence for both regression and classification: \"the probability vector obtained at the end of the pipeline (the softmax output) is often erroneously interpreted as model confidence\" (Gal, PhD thesis, p. 13)
1. Our model generalise well so we predict redshift for the whole DR16 and find QSO? (TODO)

Previously attempts to predict redshift are QuasarNet (Busca & Balland, 2018) inspired by YOLO and Stivaktakis et al. (2019) that classify into bins.
"

# ╔═╡ da123329-9d36-45cf-b743-3a333fa5bd11
md"## Methods

### Bayesian Convolutional Neural Network

We used Bayesian convolutional network (Bayesian ConvNet) to predict spectroscopic redshift.
We did it as *regression* problem, because ConvNets are able to do regression.
See YOLO: \"We reframe object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities.\" (Redmon et al., 2016).
But we also reformulated the regressin problem as classification problem and ConvNet as classification model is better.
We transformed continuous redshift by binning into ordinal categories.
Stivaktakis et al. (2019) used ConvNet as a classification model, but they do not compare to regression.

\"If the input has known topological structure (for example, if the input is an image), use a convolutional network.
In these cases, you should begin by using some kind of piecewise linear unit (ReLUs or their generalizations, such as Leaky ReLUs, PreLus, or maxout).\" (Goodfellow et al., 2016, p. 420)

\"A reasonable choice of optimization algorithm is SGD with momentum with a decaying learning rate [...]. Another reasonable alternative is Adam.
Batch normalization can have a dramatic effect on optimization performance, especially for convolutional networks and networks with sigmoidal nonlinearities.
While it is reasonable to omit batch normalization from the very first baseline, it should be introduced quickly if optimization appears to be problematic.\" (Goodfellow et al., 2016, p. 420)

\"Early stopping should be used almost universally.\" (Goodfellow et al., 2016, p. 420)

### SDSS DR12Q & DR16Q Data

Data are SDSS DR12Q and DR16Q to verify generalisation.

### Data Preparation

Data preparation consists of continuum normalisation and resampling.

### Metrics of Performance

Root-mean-square (RMS) error: $E_\mathrm{RMS} = \sqrt{\frac{1}{N} \sum_{n = 1}^N (\hat{z}_n - z_n)^2}$. (Bishop, 2006)

Velocity difference $\Delta v = c \cdot \frac{|\hat{z} - z|}{1 + z}$ and median $\Delta v$ and median absolute deviation (MAD) $\Delta v$. (Lyke et al., 2020)

Ratio of catastrophic failures: $\Delta v >  3000 \mbox{ km s}^{-1}$. (Lyke et al., 2020)

Baseline predictions and model to compare to:
- pipeline (see column `Z_PIPE` in catalogues);
- QuasarNet (column `Z_QN` in DR16Q catologue);
- [redvsblue](https://github.com/londumas/redvsblue) (`Z_PCA` in DR16Q): probably an unsupervised technique to improve given estimate (prior, i.e. `Z_PIPE`).

We can use uncertainty to refuse to make prediction of redshift.
But, we have to keep coverage high:
\"In some applications, it is possible for the machine learning system to refuse to make a decision.
This is useful when the machine learning algorithm can estimate how confident it should be about a decision, especially if a wrong decision can be harmful and if a human operator is able to occasionally take over.
[...]
A natural performance metric to use in this situaiton is **coverage**.
Coverage is the fraction of examples for wich the machine learning system is able to produce a response.
It is possible to trade coverage for accuracy.\" (Goodfellow et al., 2016, p. 419)
"

# ╔═╡ 1316241a-0a53-4db6-806c-685f20a38c7b
md"## Results"

# ╔═╡ 63738e8a-d4d0-47a4-a2a6-3990fac7463f
md"## Discussion

We filled the gap and showed that classification is better than regression.
We hypothesise that classification is better because of the cross-entropy loss why regression has mean squared error.
(See Deep Learning book: In classification mean squared error loss was replaced with cross-entropy. But, here the case is regression versus classification.)

Limitation is that we do nothing about predictions that are uncertain.
In future research, we plan to use the uncertainty in active learning to further impore predictions."

# ╔═╡ Cell order:
# ╠═5c7adecc-aefa-11eb-2bb5-f5778d7edcb2
# ╠═1a436fc9-d74a-441c-8125-1af54d17fe97
# ╠═da123329-9d36-45cf-b743-3a333fa5bd11
# ╠═1316241a-0a53-4db6-806c-685f20a38c7b
# ╠═63738e8a-d4d0-47a4-a2a6-3990fac7463f
