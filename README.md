# Bayesian Redshift

Bayesian learning to predict redshift with uncertainty.

## Data

Get data so that the result is comparable.

"The catalog consists of two subcatalogs: the DR16Q superset containing 1,440,615 observations targeted as quasars, and the quasar-only set containing 750,414 quasars.
[...]
We include automated pipeline redshifts for all quasars observed as part of SDSS-III/IV and confident visual inspection redshifts for 320,161 quasars [probably objects not spectra].
[...]
DR16Q includes homogeneous redshifts derived using principal component analysis, including six separate emission line PCA redshifts." (Lyke et al., 2020)

## Performance Metric

Goal should be to minimize the ration on catasthropic redshift. (Lyke et al., 2020)

Method | Catastrophic Redshift Ratio | Root-Mean-Square Error
------ | --------------------------- | ----------------------
`Z_PCA`                        | 0.01538 | 0.032769
ZFNet                          | 0.43328 | 0.149261
VGG11                          | 0.37538 | 0.151978
VGG16                          | 0.48056 | 0.160014
Fully-Connected Neural Network | 0.54772 | 0.193327
`Z_PIPE`                       | 0.09036 | 0.649342
Linear Regression              | 0.96170 | 0.701032
Bayesian Linear Regression     | 0.96176 | 0.701032
`Z_QN`                         | 0.13772 | 0.722877

"Determine your goals—what error metric to use, and your target value for this error metric.
These goals and error metrics should be drive by the problem that the applicatoin is intended to solve." (Goodfellow et al., 2016, p. 416)

I should derive the target value of performace metric from SDSS DR16Q paper (Lyke et al., 2020) and QuasarNet (Busca and Balland, 2018).

I can use uncertainty to refuse to make prediction of redshift.
But, I have to keep coverage high:

"In some applications, it is possible for the machine learning system to refuse to make a decision.
This is useful when the machine learning algorithm can estimate how confident it should be about a decision, especially if a wrong decision can be harmful and if a human operator is able to occasionally take over.
[...]
A natural performance metric to use in this situaiton is **coverage**.
Coverage is the fraction of examples for wich the machine learning system is able to produce a response.
It is possible to trade coverage for accuracy." (Goodfellow et al., 2016, p. 419)

## Default Baseline Models

`Z_PIPE`: Probably some kind of template matching.

`Z_QN`: See https://github.com/ngbusca/QuasarNET.

`Z_PCA`: Probably unsupervised technique to improve given estimate (prior, i.e. `Z_PIPE`). See https://github.com/londumas/redvsblue. May be use to improve my prediction.

I should use ConvNet and batch normalization (but is compatible with Bayesian approach?):

"If the input has known topological structure (for example, if the input is an image), use a convolutional network.
In these cases, you should begin by using some kind of piecewise linear unit (ReLUs or their generalizations, such as Leaky ReLUs, PreLus, or maxout)." (Goodfellow et al., 2016, p. 420)

"A reasonable choice of optimization algorithm is SGD with momentum with a decaying learning rate [...]. Another reasonable alternative is Adam.
Batch normalization can have a dramatic effect on optimization performance, especially for convolutional networks and networks with sigmoidal nonlinearities.
While it is reasonable to omit batch normalization from the very first baseline, it should be introduced quickly if optimization appears to be problematic." (Goodfellow et al., 2016, p. 420)

"Early stopping should be used almost universally." (Goodfellow et al., 2016, p. 420)

## TODO List

1. data preparation of DR16Q superset
    - ~~continuum normalisation~~
    - ~~parallelise~~
    - spectra scaling? (maybe division by max(abs(flux)))
    - ~~extract baselines: `Z_PCA`, `Z_QN`, `Z_PIPELINE`, `Z_VI` (true label)~~
    - ~~data type should be float (32 bits)~~
2. evaluation
    - ~~root-mean-square error (RMSE)~~
    - ~~as classification (catastrophic redshift, see DR16Q paper)~~
3. experiments
    - ~~linear regression~~
    - Bayesian linear regression
    - ~~fully-connected network~~
    - Bayesian neural network
    - ~~fully-connected network with dropout to get uncertainty (see Gal)~~
    - convolutional neural network
    - convolutional neural network with dropout to get uncertainty (see Gal)
    - objection detection (YOLO...)
4. loss function
    - ~~mean square error~~
    - classification into bins
5. application to the rest of DR16Q superset (without visual label)
    - domain adaptation using active learning?
6. generate imaginary data
    - "a well-defined probabilistic model, it is always possible to generate
      data from the model; such 'imaginary' data provide a window into the
      'mind' of the probabilistic model" (Ghahramani 2015)

## Installation

Install PyTorch:

    $ pip install torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

## Bibliography

Busca, N., Balland, C., 2018. QuasarNET: Human-level spectral classification and redshifting with Deep Neural Networks. arXiv:1808.09955 [astro-ph].

Goodfellow, I., Bengio, Y., Courville, A., 2016. Deep learning, Adaptive computation and machine learning. The MIT Press, Cambridge, Massachusetts.

Lyke, B.W. et al., 2020. The Sloan Digital Sky Survey Quasar Catalog: Sixteenth Data Release. ApJS 250, 8. https://doi.org/10.3847/1538-4365/aba623
