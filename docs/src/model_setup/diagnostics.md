# Diagnostics

Diagnostics are a set of general utilities that can be called on-demand during time-stepping to compute quantities of
interest you may want to save to disk, such as the horizontal average of the temperature, the maximum velocity, or to
produce a time series of salinity. They also include utilities for diagnosing model health, such as the CFL number or
to check for NaNs.

Diagnostics are stored as a list of diagnostics in `simulation.diagnostics`. Diagnostics can be specified at model creation
time or be specified at any later time and appended (or assigned with a key value pair) to `simulation.diagnostics`.

Most diagnostics can be run at specified frequencies (e.g. every 25 time steps) or specified intervals (e.g. every
15 minutes of simulation time). If you'd like to run a diagnostic on demand then do not specify any intervals
(and do not add it to `simulation.diagnostics`).
