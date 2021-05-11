### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# ╔═╡ 7ea4b94b-f8f0-4eb4-a8cf-e105e67976a6
using DataFrames, FITSIO, Statistics, StatsPlots

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

### SDSS DR12Q & DR16Q Data

Data are SDSS DR12Q superset and SDSS DR16Q superset to verify generalisation.

SDSS DR12Q catalogue is the final catalogue of QSOs from the Baryon Oscillation Spectroscopi Survey (BOSS) of SDSS-III.
The corresponding superset catalogue includes:
\"Spectra of all the SDSS-III/BOSS quasar candidates, all the SDSS-III quasar targets from ancillary programs and all the objects classified robustly as ``z \ge 2`` quasars by the SDSS pipeline (Bolton et al. 2012) among the galaxy targets (Reid et al. 2016) have been visually inspected.\"
The superset catalogue is suitable for machine learning due to the visual inspected of all spectra.
\"If the pipeline redshift does not need to be modified, the SDSS pipeline redshift (`Z_PIPE`) and the visual inspection redshift (`Z_VI`) are identical. If not, the redshift is set by eye with an accuracy that cannot be better than ``\Delta z \sim 0.003``.\"
SDSS DR12Q superset contains 546856 objects.
`Z_CONF_PERSON` refers to level of confidence for the redshift associated to each object.
Where `Z_CONF_PERSON` is equal to 3 then the redshift is not uncertain.
Final DR12Q catalogue contains a total of 297301 unique QSOs.

SDSS DR16Q superset catalogue is the final SDSS-IV QSO catalogue of extended BOSS (eBOSS).
The superset contains 1440615 spectra (observations).
There are spectra visually inspected (320161 spectra) while the rest has automated identification and redshift.
There should be 920110 observations of 750414 QSOs.

From the 546856 spectra in DR12Q superset, we selected only those with ``z > -1``, `Z_CONF_PERSON` equal to 3 and with wavelength coverage in range 3830.01–9084.48 Å (or 3.5832–3.9583 Å in logarithmic wavelengths).
Giving 523129 spectra after filtering (23727 spectra lost).

The wavelength coverage range is selected based on the SDSS DR16Q superset coverage because we want to show the generalisation power of our machine learning model and the DR16Q superset has smaller wavelength coverage.
We selected the minimal wavelength to be the 99.9 quantile (3830.01 Å) of all minimal wavelengths and the maximal wavelength to be the 0.01 quantile (9084.48 Å) of all maximal wavelength in the DR16Q catalogue.
The wavelength range in the DR16Q catalogue gives 1437742 out of 1440615 (2873 lost).
"

# ╔═╡ c4c54574-dd08-49bb-8f7b-0ffcf45a02ee
begin
	dr16q_superset_fits = FITS("data/DR16Q_Superset_v3.fits")
	dr16q_superset = DataFrame(
		plate=read(dr16q_superset_fits[2], "PLATE"),
		mjd=read(dr16q_superset_fits[2], "MJD"),
		fiberid=read(dr16q_superset_fits[2], "FIBERID"),
		z=read(dr16q_superset_fits[2], "Z"))
	dr16q_specobj_fits = FITS("data/specObj-dr16.fits")
	dr16q_specobj = DataFrame(
		plate=read(dr16q_specobj_fits[2], "PLATE"),
		mjd=read(dr16q_specobj_fits[2], "MJD"),
		fiberid=read(dr16q_specobj_fits[2], "FIBERID"),
		wavemin=read(dr16q_specobj_fits[2], "WAVEMIN"),
		wavemax=read(dr16q_specobj_fits[2], "WAVEMAX"))
	dr16q_waves = leftjoin(dr16q_superset, dr16q_specobj, on=[:plate, :mjd, :fiberid])
	dropmissing!(dr16q_waves, [:wavemin, :wavemax])
	wavemin = quantile(dr16q_waves[:wavemin], 0.999)
	wavemax = quantile(dr16q_waves[:wavemax], 0.001)
	wavemin, wavemax, log10(wavemin), log10(wavemax)
end

# ╔═╡ 059448cb-e6c5-4a6a-9dfb-f5a51c4c3dea
begin
	dr12q_superset_fits = FITS("data/Superset_DR12Q.fits")
	dr12q_superset = DataFrame(
		plate=read(dr12q_superset_fits[2], "PLATE"),
		mjd=read(dr12q_superset_fits[2], "MJD"),
		fiberid=read(dr12q_superset_fits[2], "FIBERID"),
		z_vi=read(dr12q_superset_fits[2], "Z_VI"),
		z_conf_person=read(dr12q_superset_fits[2], "Z_CONF_PERSON"))
	dr12q_specobj_fits = FITS("data/specObj-dr12.fits")
	dr12q_specobj = DataFrame(
		plate=read(dr12q_specobj_fits[2], "PLATE"),
		mjd=read(dr12q_specobj_fits[2], "MJD"),
		fiberid=read(dr12q_specobj_fits[2], "FIBERID"),
		wavemin=read(dr12q_specobj_fits[2], "WAVEMIN"),
		wavemax=read(dr12q_specobj_fits[2], "WAVEMAX"))
	dr12q_gt_minus_one_idx = -1.0 .< dr12q_superset[:z_vi]
	dr12q_z_conf_idx = dr12q_superset[:z_conf_person] .== 3
	dr12q_waves = leftjoin(dr12q_superset, dr12q_specobj, on=[:plate, :mjd, :fiberid])
	dr12q_wave_idx = ((dr12q_waves[:wavemin] .<= 10 ^ 3.5832)
		.& (10 ^ 3.9583 .<= dr12q_waves[:wavemax]))
	dr12q_subset = dr12q_waves[
		dr12q_gt_minus_one_idx .& dr12q_z_conf_idx .& dr12q_wave_idx, :]
	size(dr12q_superset, 1), size(dr12q_subset, 1)
end

# ╔═╡ 156df393-e44c-4981-b4fb-e56a83e0292a
begin
	dr16q_wave_idx = ((dr16q_waves[:wavemin] .<= wavemin)
		.& (dr16q_waves[:wavemax] .>= wavemax))
	dr16q_subset = dr16q_waves[dr16q_wave_idx, :]
	size(dr16q_superset, 1), size(dr16q_subset, 1)
end

# ╔═╡ a77d68ac-6d47-468a-a00f-52ff8df61d72
begin
	@df dr12q_subset histogram(
		:z_vi, xlabel="z", ylabel="Count", label="Z_VI in DR12Q Superset")
	@df dr16q_superset[dr16q_superset[:z] .> -1, :] histogram!(
		:z, label="Z in DR16Q Superset")
end

# ╔═╡ ada2d063-1629-4ce0-a28c-ece36bf7f41b
md"### Data Preparation

Data preparation consists of continuum normalisation, resampling and split of data sets.

We get each spectrum from the individual FITS files.
To do the continuum normalisation, we firstly standardise the fluxes to ensure numerical stability:
```math
\mathbf{x}' = \frac{\mathbf{x} - \bar{\mathbf{x}}}{σ(\mathbf{x})}.
```
Then, we applied the density of the least squares (DLS) method (Bukvić et al. 2006) with the third order polynomial (not using the inverse variances of each flux) and we substracted the continuum from the standardised spectrum.
Standardisation and continuum normalisation make spectra invariant to scale, intensities and continuum shape, i.e. we focus mainly on spectral lines.

!!! warning \"TODO: Continuum Normalisation\"

    Experiment without continuum normalisation to se if it help else remove it.

After continuum normalisation, we resampled each spectrum using the SpectRes algorithm (Carnall 2017).
The original wavelength range is from 3.5832 to 3.9583 Å with 3752 measurements.
The new wavelength range is from 3.5842 to 3.9583 Å (we added resp. subtracted 0.005 to avoid `null` value on edges due to intrinsic properties of the algorithm.
We experimented with different number of measument in the new wavelength range (128, 256 and 512).

Finally, we split the DR12Q data into training, validation, and test sets.
With inspiration from sizes of splits of ImageNet Large Scale Visual Recognition Competition (Russakovsky et al. 2014), sizes of our validation and test sets are 50000 spectra.
The remaining spectra are in the training set (423129 spectra).
We do not need to split the DR16Q data because they serve only for evaluation purposes.

The final design matrixes and output vectors are floating point numbers with 32 bits.
"

# ╔═╡ 0fc6e6ba-d8f1-40df-8bab-6d0dc58f0b83
# TODO show data preparation in a figure

# ╔═╡ 1d9c8bd4-3401-47b7-87dc-69bbfccc2fe4
md"### Bayesian Convolutional Neural Network

We used Bayesian convolutional network (Bayesian ConvNet) to predict spectroscopic redshift.
We did it as *regression* problem, because ConvNets are able to do regression.
See YOLO: \"We reframe object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities.\" (Redmon et al. 2016).
But we also reformulated the regressin problem as classification problem and ConvNet as classification model is better.
We transformed continuous redshift by binning into ordinal categories.
Stivaktakis et al. (2019) used ConvNet as a classification model, but they do not compare to regression.

Architercture:
- convolutional neural network (ConvNet):
  \"If the input has known topological structure (for example, if the input is an image), use a convolutional network.\" (Goodfellow et al. 2016, p. 420);
- ReLUs activation functions (``\max(0, z)``):
  \"In these cases, you should begin by using some kind of piecewise linear unit (ReLUs or their generalizations, such as Leaky ReLUs, PreLus, or maxout).\" (Goodfellow et al. 2016, p. 420);
- same padding.

Optimisation:
- Adam optimiser (learning rate ``\eta = 0.001``, ``\beta_1 = 0.9``, and ``\beta_2 = 0.999``) (Kingma & Ba 2017):
  \"A reasonable choice of optimization algorithm is SGD with momentum with a decaying learning rate [...]. Another reasonable alternative is Adam.\" (Goodfellow et al. 2016, p. 420);
- batch size: 256;
- early stopping with patience of 32 epochs:
  we stop training if no improvement in cat. ``z`` ratio during last 32 epochs,
  32 epochs is a trade-off between a convergence and duration of training,
  we recover the best model found
  (Goodfellow et al. 2016, p. 420: \"Early stopping should be used almost universally.\");
- regression uses the *mean squared error* loss;
- classification uses the *cross-entropy* loss;
- bayesian models use weight decay ``\lambda``:
  we used grid search to set the ``\lambda`` hyperparameter.
"

# ╔═╡ ce257413-8fe8-4bcb-b317-18af463fbd2b
md"### Metrics of Performance

1. Root-mean-square (RMS) error: ``E_\mathrm{RMS} = \sqrt{\frac{1}{N} \sum_{n = 1}^N (\hat{z}_n - z_n)^2}``. (Bishop 2006)
1. Velocity difference ``\Delta v = c \cdot \frac{|\hat{z} - z|}{1 + z}`` and median ``\Delta v`` and median absolute deviation (MAD) ``\Delta v``. (Lyke et al. 2020)
1. Ratio of catastrophic failures: ``\Delta v > 3000 \mbox{ km s}^{-1}``. (Lyke et al. 2020)

Baseline predictions and model to compare to:
- pipeline (see column `Z_PIPE` in catalogues and Bolton et al. 2012);
- QuasarNet (column `Z_QN` in DR16Q catologue);
- [redvsblue](https://github.com/londumas/redvsblue) (`Z_PCA` in DR16Q): probably an unsupervised technique to improve given estimate (prior, i.e. `Z_PIPE`).

We can use uncertainty to refuse to make prediction of redshift.
But, we have to keep coverage high:
\"In some applications, it is possible for the machine learning system to refuse to make a decision.
This is useful when the machine learning algorithm can estimate how confident it should be about a decision, especially if a wrong decision can be harmful and if a human operator is able to occasionally take over.
[...]
A natural performance metric to use in this situaiton is **coverage**.
Coverage is the fraction of examples for wich the machine learning system is able to produce a response.
It is possible to trade coverage for accuracy.\" (Goodfellow et al. 2016, p. 419)
"

# ╔═╡ 1316241a-0a53-4db6-806c-685f20a38c7b
md"## Results

Hyperparameter evaluatoin on DR12Q superset validation set:

| Experimental Setup | ``E_\mathrm{RMS}`` | Median ``\Delta v`` | Cat. ``z`` (%) |
|:-------------------| ------------------:| -------------------:| --------------:|
| Input 256 (Reg)    | TODO               | TODO                | TODO           |
| Input 256 (Clf)    | TODO               | TODO                | TODO           |
| Input 128 (Clf)    | TODO               | TODO                | TODO           |
| Input 512 (Clf)    | TODO               | TODO                | TODO           |
| ``\lambda = ?``    | TODO               | TODO                | TODO           |

Final evaluation on DR12Q superset test set
(baselines and model with hyperparameters optimised on DR12Q validation set):

| Model                 | ``E_\mathrm{RMS}`` | Median ``\Delta v`` | Cat. ``z`` (%) |
|:----------------------| ------------------:| -------------------:| --------------:|
| `Z_PIPE`              | TODO               | TODO (TODO)         | TODO           |
| `Z_PCA`               | TODO               | TODO (TODO)         | TODO           |
| `Z_QN` (DR16Q)        | TODO               | TODO (TODO)         | TODO           |
| Reg ConvNet           | TODO               | TODO (TODO)         | TODO           |
| Bayesian Reg ConvNet  | TODO               | TODO (TODO)         | TODO           |
| Clf ConvNet           | TODO               | TODO (TODO)         | TODO           |

Evaluation on DR16Q superset is tricky because not all spectra are visually inspected.
"

# ╔═╡ 63738e8a-d4d0-47a4-a2a6-3990fac7463f
md"## Discussion

We filled the gap and showed that classification is better than regression.
We hypothesise that classification is better because of the cross-entropy loss why regression has mean squared error.
(See Deep Learning book: In classification mean squared error loss was replaced with cross-entropy. But, here the case is regression versus classification.)

Limitation is that we do nothing about predictions that are uncertain.
In future research, we plan to use the uncertainty in active learning to further impore predictions."

# ╔═╡ Cell order:
# ╟─5c7adecc-aefa-11eb-2bb5-f5778d7edcb2
# ╠═7ea4b94b-f8f0-4eb4-a8cf-e105e67976a6
# ╟─1a436fc9-d74a-441c-8125-1af54d17fe97
# ╟─da123329-9d36-45cf-b743-3a333fa5bd11
# ╠═c4c54574-dd08-49bb-8f7b-0ffcf45a02ee
# ╠═059448cb-e6c5-4a6a-9dfb-f5a51c4c3dea
# ╠═156df393-e44c-4981-b4fb-e56a83e0292a
# ╠═a77d68ac-6d47-468a-a00f-52ff8df61d72
# ╟─ada2d063-1629-4ce0-a28c-ece36bf7f41b
# ╠═0fc6e6ba-d8f1-40df-8bab-6d0dc58f0b83
# ╟─1d9c8bd4-3401-47b7-87dc-69bbfccc2fe4
# ╟─ce257413-8fe8-4bcb-b317-18af463fbd2b
# ╠═1316241a-0a53-4db6-806c-685f20a38c7b
# ╟─63738e8a-d4d0-47a4-a2a6-3990fac7463f
