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
date: 17 December 2019
bibliography: paper.bib
---

# Summary

``Oceananigans.jl`` is a fast and friendly software package for the numerical
simulation of incompressible, stratified, rotating fluid flows on CPUs and GPUs.
It is being developed as part of the Climate Modeling Alliance project for the
simulation of small-scale ocean physics at high-resolution that affect the
evolution of Earth’s climate.

``Oceananigans.jl`` is designed for high-resolution simulations in idealized
geometries and supports direct numerical simulation, large eddy simulation,
arbitrary numbers of active and passive tracers, and linear and nonlinear
equations of state for seawater. Under the hood, Oceananigans.jl employs a
finite volume algorithm similar to that used by the Massachusetts Institute of
Technology general circulation model [@Marshall1997].

![(Left) Large eddy simulation of small-scale oceanic boundary layer turbulence forced by a surface cooling in a horizontally periodic domain using $256^3$ grid points. The upper layer is well-mixed by turbulent convection and bounded below by a strong buoyancy interface. (Right) Simulation of instability of a horizontal density gradient in a rotating channel using $256\times512\times128$ grid points. A similar process called baroclinic instability acting on basin-scale temperature gradients fills the ocean with eddies that stir carbon and heat.](free_convection_and_baroclinic_instability.png)

``Oceananigans.jl`` leverages the Julia programming language [@Bezanson2017] to
implement high-level, low-cost abstractions, a friendly user interface, a
high-performance model in one language and a common code base for execution on
the CPU or GPU with Julia’s native GPU compiler [@Besard2019]. Because Julia is
a high-level language, development is faster and users can flexibly specify
model configurations, set up arbitrary diagnostics and output, extend the code
base, and implement new features.  Configuring a model with `architecture=CPU()`
or `architecture=GPU()` will execute the model on the CPU or GPU.

Performance benchmarks show significant speedups when running on a GPU. Large
simulations on an Nvidia Tesla V100 GPU require ~1 nanosecond per grid point per
iteration. This results in GPU simulations being roughly 3x more cost-effective
than CPU simulations on cloud computing platforms such as Google Cloud. A GPU
with 32 GB of memory can time-step models with ~150 million grid points assuming
five fields are being evolved; for example, three velocity components, and
tracers for temperature and salinity. These performance gains permit the
long-time integration of realistic simulations, such as large eddy simulation of
oceanic boundary layer turbulence over a seasonal cycle or the generation of
training data for turbulence parameterizations in Earth system models.

``Oceananigans.jl`` is continuously tested with unit tests, integration tests,
analytic solutions to the incompressible Navier-Stokes equations, and
verification experiments against published scientific results. The verification
experiments also serve as documented and advanced examples. Future development
plans include support for distributed parallelism with CUDA-aware MPI as well as
bathymetry and irregular domains.

# Acknowledgements

Our work is supported by the generosity of Eric and Wendy Schmidt by
recommendation of the Schmidt Futures program, and by the National Science
Foundation under grant AGS-6939393.

# References
