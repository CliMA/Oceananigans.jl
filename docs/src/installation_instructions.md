# Installation instructions
You can install the latest version of Oceananigans using the built-in package manager (accessed by pressing `]` in the
Julia command prompt) to add the package and instantiate/build all dependencies
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

!!! warn "Use Julia 1.1 or newer"
    Oceananigans requires at least Julia v1.1 to run correctly.

    If you try to install Oceananigans with Julia v1.0 then it will install a very old version of Oceananigans (the
    latest version that is compatible with v1.0).
