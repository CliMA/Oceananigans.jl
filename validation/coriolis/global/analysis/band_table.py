"""Band variances of surface v (cm²/s²) at equal days for the runs from rest.
Usage: python3 band_table.py [days, comma separated] [float type]"""
import sys
from bands import bands, band_variance
from readjld import read, iterations

days = [int(d) for d in sys.argv[1].split(",")] if len(sys.argv) > 1 else [30, 60, 90, 120, 180, 240, 300, 360]
float_type = sys.argv[2] if len(sys.argv) > 2 else "Float64"
cases = {name: f"{name}_{float_type}" for name in ("FourPoint", "CD", "FixedSE", "FixedNW", "ShearSigned")}
available = {name: set(iterations(case)) for name, case in cases.items()}
available = {name: its for name, its in available.items() if its}
for band in bands:
    print(f"{band}".ljust(22) + "".join(f"{d:7d}" for d in days))
    for name, its in available.items():
        print(f"   {name:19s}" + "".join(f"{band_variance(read(cases[name], 'v', 24 * d), band):7.3f}" if 24 * d in its else f"{'-':>7s}" for d in days))
