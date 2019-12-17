---
title: 'Oceananigans.jl: Fast and friendly geophysical fluid dynamics on GPUs'
tags:
  - Julia
  - fluid
  - ocean
  - climate
  - gpu
authors:
  - name: Ali Ramadhan
    orcid: 0000-0003-1102-1520
    affiliation: 1
  - name: Gregory LeClaire Wagner
    affiliation: 1
  - name: Chris Hill
    affiliation: 1
  - name: Jean-Michel Campin
    affiliation: 1
  - name: Valentin Churavy
    affiliation: 2
  - name: Tim Besard
    affiliation: 2
  - name: Andre Souza
    affiliation: 1
  - name: Alan Edelman
    affiliation: 1
  - name: John Marshall
    affiliation: 1
  - name: Raffaele Ferrari
    affiliation: 1
affiliations:
 - name: Massachusetts Institute of Technology
   index: 1
 - name: Julia Computing, Inc.
   index: 2
date: 16 December 2019
bibliography: paper.bib
---

# Summary

``Oceananigans.jl`` is a fast and friendly software package for the numerical
simulation of incompressible, stratified, rotating fluids on CPUs and GPUs.
It is being developed as part of the Climate Modeling Alliance project for the
simulation of small-scale ocean physics at high-resolution that affect the
evolution of Earth’s climate.

``Oceananigans.jl`` is designed for high-resolution simulations in idealized
geometries and supports direct numerical simulation, large eddy simulation,
arbitrary numbers of active and passive tracers, and linear and nonlinear
equations of state for seawater. Under the hood, Oceananigans.jl employs a
finite volume algorithm similar to that used by the Massachusetts Institute of
Technology general circulation model [@Marshall1997].

``Oceananigans.jl`` leverages the Julia programming language [@Bezanson2017] to
implement high-level, low-cost abstractions, a friendly user interface, and a
high-performance model in one language. Julia, being a high-level language,
greatly reduces development time and allows users to more easily extend the
model and implement new features.

Julia’s functional programming paradigm enables arbitrary spatially-varying and
time-dependent forcing functions and boundary conditions to be defined as
functions and passed to the model. This simplifies the implementation of
complicated yet commonly-used features such as radiation and open boundary
conditions and setting up a model linearized about a base state. Arbitrary
quantities such as vorticity, turbulent kinetic energy, and advective fluxes
can be diagnosed on demand. More general quantities such as horizontal and time
averages, field maxima, and time series can be diagnosed on the fly as well.

Using Julia's native GPU compiler [@Besard2019], we develop a single code base
that compiles and executes efficiently on CPUs and GPUs. Writing
``architecture=GPU()`` instead of ``architecture=CPU()`` when configuring a
model specifies it to execute on the GPU. Performance benchmarks show
significant speedups when running on a GPU. Large simulations on an Nvidia
Tesla V100 GPU require only ~1 nanosecond per grid point per iteration. This
also results in GPU simulations being roughly 3x more cost-effective than CPU
simulations on cloud computing platforms such as Google Cloud. These
performance gains allow for the long-time integration of demanding simulations
that fit on a single GPU, such as large eddy simulation of oceanic boundary
layer turbulence over a seasonal cycle and, for example, the generation of
training data with huge ensembles of simulations where each ensemble member
fits on a single GPU.

``Oceananigans.jl`` is continuously tested with unit tests, integration tests,
analytic solutions to the incompressible Navier-Stokes equations, and
verification experiments against published scientific results. The verification
experiments also serve as documented and advanced examples.

Future development work includes distributed parallelism capabilities to allow
for much larger simulations on multiple CPUs and multiple GPUs with CUDA-aware
MPI, the addition of higher-order advection schemes, and support for topography.

``Oceananigans.jl`` makes sophisticated science easy, makes scientists more
productive, and makes high-powered computational fluid dynamics and ocean
modeling accessible to students, scientists, and users with a variety of
backgrounds. Oceananigans.jl is flexible, extensible, and can be used
interactively or through scripts.

# Acknowledgements

Our work is supported by the generosity of Eric and Wendy Schmidt by
recommendation of the Schmidt Futures program, and by the National Science
Foundation under grant AGS-6939393.

# References
