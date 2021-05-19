# Installation instructions

You can install the latest version of Oceananigans using the built-in package manager (accessed by pressing `]` in the
Julia command prompt) to add the package and instantiate/build all dependencies

```julia
julia>]
(v1.6) pkg> add Oceananigans
(v1.6) pkg> instantiate
```

We recommend installing Oceananigans with the built-in Julia package manager, because this installs a stable, tagged
release. Oceananigans.jl can be updated to the latest tagged release from the package manager by typing

```julia
(v1.6) pkg> update Oceananigans
```

At this time, updating should be done with care, as Oceananigans is under rapid development and breaking changes to the user API occur often. But if anything does happen, please open an issue!

But if anything does happen or your code stops working, please open an issue and ask! We're more than happy to help with getting your simulations up and running.

!!! warn "Use Julia 1.6 or newer"
    The latest version of Oceananigans requires at least Julia v1.6 to run.
    Installing Oceananigans with an older version of Julia will install an older version of Oceananigans (the latest version compatible with your version of Julia).
