# Spectroscopic redshift prediction with Bayesian convolutional networks

Current astronomy is facing vast amounts of data that will probably never be visually inspected.
If there is a lot of labelled data, convolutional neural networks (that belong to the ﬁeld of deep learning) are a powerful tool for any task.
We introduce a method for spectroscopic redshift prediction based on a Bayesian convolutional neural network inspired by VGG networks.
We empirically prove that formulating redshift prediction as a classiﬁcation task by binning is better than regressing the redshift.
The advantage of the method over contemporary methods for spectroscopic redshift prediction is that it provides a predictive uncertainty.
The predictive uncertainty enables us to
(1.) determine unusual or problematic spectra for visual inspection
(additionally, the Bayesian network can support the inspection with suggestions of probable redshift);
(2.) do thresholding that allows us to balance between the number of incorrect redshift predictions and coverage.
We used the 12th Sloan Digital Sky Survey quasar superset as the training set for the method,
and we evaluated its generalisation capability on about three-quarters of a million spectra from the 16th quasar superset of the same survey.
On the 16th quasar superset, the method performs equally well as the most used template ﬁtting method but additionally provides the predictive uncertainty.

## List of Files

- `data/`: data directory
- `models/`: trained models
- `slurm/`: directory with batch scripts for slurm workload manager
- `Evaluation.jl`: code for evaluation
- `Neural.jl`: code for convolutional neural networks
- `Utils.jl`: utility code
- `dr12q_superset_eval.jl`: evaluation on the DR12Q superset
- `dr12q_superset_expl.jl`: exploration of the DR12Q superset
- `dr12q_superset_extr.jl`: data preparation of the DR12Q superset
- `dr16q_superset_entr.jl`: standard entropy from the classical convolutional neural network
- `dr16q_superset_eval.jl`: evaluation on the DR16Q superset
- `dr16q_superset_expl.jl`: exploration of the DR16Q superset
- `dr16q_superset_extr.jl`: data preparation of the DR16Q superset
- `dr16q_superset_unce.jl`: uncertainty from the Bayesian convolutional neural network
