import h5py
import numpy as np
from spectres import spectres


LOGLAMMIN, LOGLAMMAX = 3.5832, 3.9583
N_FEATURES = 256
EPS = 0.005


with h5py.File("data/dr16q_superset.hdf5", "r+") as datafile:
    fluxes = datafile["flux"][:]

    # resample
    n_waves = fluxes.shape[1]
    loglam = np.linspace(LOGLAMMIN, LOGLAMMAX, n_waves)
    # EPS else will get nans in output
    new_loglam = np.linspace(LOGLAMMIN + EPS, LOGLAMMAX - EPS, N_FEATURES)
    X = spectres(
            new_loglam, loglam, fluxes,
            verbose=True).astype(np.float32, copy=False)
    X_dset = datafile.create_dataset("X", data=X)
