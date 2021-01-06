# Bayesian Redshift

Bayesian learning to predict redshift with uncertainty.

## TODO List

1. data preparation of DR16Q superset
    - ~~continuum normalisation~~
    - parallelise
    - standardisation (maybe division by max(abs(flux)))
    - ~~extract baselines `Z_PCA`, `Z_QN`, `Z_PIPELINE`, `Z_VI` (true label)...~~
    - ~~data type should be float (32 bits)~~
2. evaluation
    - ~~root-mean-square error (RMSE)~~
    - ~~as classification? (catastrophic redshift, see DR16Q paper)~~
3. experiments
    - ~~linear regression~~
    - ~~Bayesian linear regression~~
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
