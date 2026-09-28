"""Surface v variance (cm²/s²) in zonal wavenumber bands K = kΔ, from windowed zonal Fourier transforms over all open-ocean
segments of 64 cells. Trade bands: K = 0.4–0.7π (wavelengths 2.9–5Δ); near-Nyquist: K = 0.7–1π; mesoscale: K = 0.1–0.4π."""
import numpy as np
from readjld import coordinates

λ, φ = coordinates()
latitude = φ[100, :]

bands = {"8-22S": ((-22, -8), (0.4, 0.7)), "8-22N": ((8, 22), (0.4, 0.7)), "30-50S": ((-50, -30), (0.7, 1.01)),
         "30-50N": ((30, 50), (0.7, 1.01)), "30-50N mesoscale": ((30, 50), (0.1, 0.4)), "55-65S mesoscale": ((-65, -55), (0.1, 0.4))}
n = 64
window = np.hanning(n)
K = 2 * np.pi * np.fft.rfftfreq(n)

def segments(row):
    wet = row != 0
    for start in range(0, 360 - n + 1, n // 2):
        if wet[start:start + n].all():
            yield row[start:start + n]

def band_variance(v, band):
    (south, north), (lowest, highest) = bands[band]
    selected = (K >= lowest * np.pi) & (K < highest * np.pi)
    total, count = 0.0, 0
    for j in np.where((latitude >= south) & (latitude < north))[0]:
        for segment in segments(v[:, j]):
            spectrum = np.abs(np.fft.rfft((segment - segment.mean()) * window))**2 / (np.sum(window**2) / 2)
            total += spectrum[selected].sum() / n
            count += 1
    return 1e4 * total / max(count, 1)
