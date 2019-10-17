# Oceananigans.jl
*A fast and friendly incompressible fluid flow solver in Julia that can be run in 1-3 dimensions on CPUs and GPUs.*

Oceananigans.jl is a fast and friendly incompressible fluid flow solver written in Julia that can be run in 1-3
dimensions on CPUs and GPUs. It is designed to solve the rotating Boussinesq equations used in non-hydrostatic ocean
modeling but can be used to solve for any incompressible flow.

Our goal is to develop a friendly and intuitive package allowing users to focus on the science. Thanks to high-level,
zero-cost abstractions that the Julia programming language makes possible, the model can have the same look and feel no
matter the dimension or grid of the underlying simulation, and the same code is shared between the CPU and GPU.

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

