# Oceananigans.jl
*A fast and friendly incompressible fluid flow solver in Julia that can be run in 1-3 dimensions on CPUs and GPUs.*

Oceananigans.jl is a fast and friendly incompressible fluid flow solver written in Julia that can be run in 1-3
dimensions on CPUs and GPUs. It simulates the rotating Boussinesq equations in rectangular domains with some 
special features for fluids stratified by both temperature and salinity (oceans!) --- but can also be used without
rotation, stratification, with aribtrary tracers, and arbitrary user-defined forcing functions.

We strive for a user interface that makes `Oceananigans.jl` as friendly and intuitive to use as possible, 
allowing users to focus on the science. Internally, we have attempted to write the underlying algorithm
so that the code runs as fast as possible for the configuration chosen by the user --- from simple
two-dimensional setups to complex three-dimensional simulations --- and so that as much code
as possible is shared between the CPU and GPU algorithms.

## Installation instructions
You can install the latest version of Oceananigans using the built-in package manager (accessed by pressing `]` in the
Julia command prompt) to add the package and instantiate/build all depdendencies
```julia
julia>]
(v1.1) pkg> add Oceananigans
(v1.1) pkg> instantiate
```
We recommend installing Oceananigans with the built-in Julia package manager, because this installs a stable, tagged
release. Oceananigans.jl can be updated to the latest tagged release from the package manager by typing
```julia
(v1.1) pkg> update Oceananigans
```
At this time, updating should be done with care, as Oceananigans is under rapid development and breaking changes to the
user API occur often. But if anything does happen, please open an issue!

**Note**: Oceananigans requires at least Julia v1.1 to run correctly.

## Getting help
If you are interested in using Oceananigans.jl or are trying to figure out how to use it, please feel free to ask us
questions and get in touch! Check out the
[examples](https://github.com/climate-machine/Oceananigans.jl/tree/master/examples)
and
[open an issue](https://github.com/climate-machine/Oceananigans.jl/issues/new)
if you have any questions, comments, suggestions, etc.

