# Oceananigans.jl

*A fast and friendly incompressible fluid flow solver in Julia that can be run in 1-3 dimensions on CPUs and GPUs.*

Oceananigans.jl is a fast and friendly incompressible fluid flow solver written in Julia that can be run in 1-3
dimensions on CPUs and GPUs. It simulates the rotating Boussinesq equations in rectangular domains with some
special features for fluids stratified by both temperature and salinity (oceans!) --- but can also be used without
rotation, stratification, with arbitrary tracers, and arbitrary user-defined forcing functions.

We strive for a user interface that makes `Oceananigans.jl` as friendly and intuitive to use as possible,
allowing users to focus on the science. Internally, we have attempted to write the underlying algorithm
so that the code runs as fast as possible for the configuration chosen by the user --- from simple
two-dimensional setups to complex three-dimensional simulations --- and so that as much code
as possible is shared between the CPU and GPU algorithms.

## Getting help

If you are interested in using Oceananigans.jl or are trying to figure out how to use it, please feel free to ask us
questions and get in touch! If you're trying to set up a model then check out the examples and model setup
documentation. Please feel free to [open an issue](https://github.com/CLiMA/Oceananigans.jl/issues/new)
if you have any questions, comments, suggestions, etc!
