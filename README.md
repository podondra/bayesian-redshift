# Spectroscopic redshift prediction with Bayesian convolutional networks

Current astronomy is facing vast amounts of data that will probably never be visually inspected.
If there is a lot of labelled data, convolutional neural networks (that fall into the field of deep learning) are a powerful tool for any task.
We introduce a Bayesian convolutional neural network inspired by VGG networks for spectroscopic redshift prediction.
The network is trained on the 12th Sloan Digital Sky Survey quasar superset,
and its generalisation capability is evaluated on about three-quarters of a million spectra from the 16th quasar superset of the same survey.
We empirically prove that formulating redshift prediction as a classification task by binning is better than regressing the redshift.
On the 16th quasar superset, the classification Bayesian network a achieves catastrophic *z* ratio of about 2.63 per cent and median *Δv* = 332.25 km/s performing the same as the most used template fitting method.
The advantage of the Bayesian network over the template fitting method and classical neural networks is that it provides uncertainty.
We estimated the uncertainty in the form of entropy that captures both the predictive and model uncertainty.
The entropy enables us to
(1.) determine unusual or problematic spectra for visual inspection (additionally, the Bayesian network can support the inspection with redshift suggestions);
(2.) do thresholding that can significantly reduce the number of incorrect redshift predictions.
For example, if we exclude 10 per cent of the most uncertain predictions, only 1.03 per cent of redshifts are predicted incorrectly.

## List of Files

- `data/`: data directory
- `models/`
- `slurm/`: directory with batch scripts for slurm workload manager
- `Evaluation.jl`
- `Neural.jl`
- `Utils.jl`
- `dr12q_superset_eval.jl`
- `dr12q_superset_expl.jl`
- `dr12q_superset_extr.jl`
- `dr16q_superset_entr.jl`: standard entropy from the classical convolutional neural network
- `dr16q_superset_eval.jl`
- `dr16q_superset_expl.jl`
- `dr16q_superset_extr.jl`
- `dr16q_superset_unce.jl`: uncertainty from the Bayesian convolutional neural network

## TODO list

1. Update abstract.
1. Provide a list of files with description.
