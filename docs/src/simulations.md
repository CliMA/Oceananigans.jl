# Simulations

`Simulation`s are basically a utility for managing a time-stepping loop, including scheduling
important activities such as:

* Logging the progress of a simulation,
* Computing and writing diagnostics or other output to disk,
* Stopping a simulation when its time has come.

The most important line in any script is `run!(simulation)`.
