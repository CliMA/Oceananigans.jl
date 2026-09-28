"""Surface snapshots written by setup.jl (Float32, one compressed chunk per snapshot) and the grid coordinates written by
run_global.jl. The output directory is $CORIOLIS_OUTPUT, or ../output. Iterations are hours (Δt = 1 hour)."""
import os, zlib
import h5py, numpy as np

output = os.environ.get("CORIOLIS_OUTPUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output"))
H = 5
Nx, Ny = 360, 180

def path(case):
    return os.path.join(output, f"{case}.jld2")

def read(case, field, iteration):
    with h5py.File(path(case), "r") as f:
        d = f[f"timeseries/{field}/{iteration}"]
        mask, raw = d.id.read_direct_chunk((0,) * len(d.shape))
        a = np.frombuffer(zlib.decompress(raw), dtype="<f4").reshape(d.shape)[0].T
    return a[H:H+Nx, H:H+Ny].astype(np.float64)

def iterations(case, field="v"):
    if not os.path.exists(path(case)):
        return []
    with h5py.File(path(case), "r") as f:
        return sorted(int(k) for k in f[f"timeseries/{field}"].keys() if k.isdigit())

def coordinates():
    """Longitude and latitude of the cell centres, arrays of shape (Nx, Ny)."""
    data = np.loadtxt(os.path.join(output, "coordinates.txt")).reshape(Ny, Nx, 2)
    return data[..., 0].T, data[..., 1].T
