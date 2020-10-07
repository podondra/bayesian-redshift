from math import ceil

from astropy.io import fits
import h5py
import numpy as np
from mpi4py import MPI


N_WAVES = 3826
LOGLAMMIN, LOGLAMMAX = 3.5812, 3.9637

BASE = "/home/podszond/redshift/"
TEMPLATE = "data/DR16Q_Superset_v3/{:04d}/spec-{:04d}-{:05d}-{:04d}.fits"


comm = MPI.COMM_WORLD    # communicator which links all our processes together
rank = comm.Get_rank()    # number which identifies this process
size = comm.Get_size()    # number of processes in a communicator

with h5py.File("dr16q_superset.hdf5", 'r+', driver="mpio", comm=comm) as datafile:
    ids = datafile["id"][:]

    # divide data between processes
    n = ids.shape[0]
    chunk = ceil(n / size)
    start = rank * chunk
    end = start + chunk if start + chunk <= n else n

    flux_dset = datafile.create_dataset("flux", shape=(n, N_WAVES), dtype=np.float32)

    for i in range(start, end):
        plate, mjd, fiberid = ids["plate"][i], ids["mjd"][i], ids["fiberid"][i]
        filepath = TEMPLATE.format(plate, plate, mjd, fiberid)
        with fits.open(BASE + filepath) as hdulist:
            data = hdulist[1].data
            loglam = data["loglam"]
            flux = data["flux"][(LOGLAMMIN <= loglam) & (loglam <= LOGLAMMAX)]
            flux_dset[i] = flux
