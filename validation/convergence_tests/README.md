# Convergence Tests

This directory contains scripts and modules for testing the numerical
convergence of `Oceananigans` time stepping algorithms and spatial discretiation.

To instantiate the convergence test environment, run

```
julia -e 'using Pkg; Pkg.activate(pwd()); Pkg.instantiate(); Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..", "..")))'
```

## Time stepping convergence tests

```
julia --project point_exponential_decay.jl
```

produces `figs/point_exponential_decay_time_stepper_convergence.png`.

## One-dimensional advection-diffusion tests

### Advection and diffusion of a cosine

```
julia --project one_dimensional_cosine_advection_diffusion.jl
```

produces

* `figs/cosine_advection_diffusion_solutions.png`
* `figs/cosine_advection_diffusion_error_convergence.png`

### Advection and diffusion of a Gaussian

```
julia --project one_dimensional_gaussian_advection_diffusion.jl
```

produces

* `figs/gaussian_advection_diffusion_solutions.png`
* `figs/gaussian_advection_diffusion_error_convergence.png`

## Two-dimensional diffusion

```
julia --project two_dimensional_diffusion.jl
```

produces `figs/two_dimensional_diffusion_convergence.png`.

## Two-dimensional Taylor-Green vortex

```
julia --project run_taylor_green.jl
```

and then

```
julia --project analyze_taylor_green.jl
```

produces `figs/taylor_green_convergence.png`.

## Two-dimensional forced flow with free-slip boundary conditions

```
julia --project run_forced_free_slip.jl
```

followed by

```
julia --project analyze_forced_free_slip.jl
```

produces `figs/forced_free_slip_convergence.png`.

## Two-dimensional forced flow with fixed-slip boundary conditions

```
julia --project run_forced_fixed_slip.jl
```

followed by

```
julia --project analyze_forced_fixed_slip.jl
```

produces `figs/forced_fixed_slip_convergence.png`.
