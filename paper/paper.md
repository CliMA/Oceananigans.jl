---
title: 'Oceananigans.jl: A fast and friendly fluid flow solver'
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
  - name: Andre Souza
    affiliation: 1
  - name: John Marshall
    affiliation: 1
  - name: Raffaele Ferrari
    affiliation: 1
affiliations:
 - name: Department of Earth, Atmospheric, and Planetary Sciences, Massachusetts Institute of Technology
   index: 1
date: 12 September 2019
bibliography: paper.bib
---

# Summary

``Oceananigans`` is a fast and friendly fluid flow solver written in Julia that
can be run in 1-3 dimensions on CPUs and GPUs. Designed with high-resolution
simulation of idealized geometries in mind, it employs a similar algorithm to
the the Massachusetts Institute of Technology general circulation model
[@Marshall1997] and ships with large eddy simulation capabilities.

Using Julia [@Bezanson2017] has allowed for the development of high-level,
low-cost abstractions so that both a friendly user interface and a
high-performance model can be programmed in the same language. Using a
high-level language greatly reduces development time and allows users to easily
extend the model and implement new features to carry out their experiments.
Furthermore Julia's native GPU compiler [@Besard2019] allowed us to develop a
single code base that executes efficiently on CPUs and GPUs.

Third paragraph on nice features and aspects:
1. Makes computational fluid dynamics and ocean modeling more accessible:
   existing models generally not very user-friendly as you have to fumble
   with compilers and MPI. Also written in Fortran which is no longer
   being taught to students.
2. Makes LES more accessible: existing LES code is not usually shared,
   tested, or easy to use.
3. Simulations are set up by scripts: extensible and flexible, no longer
   limited by namelist or configuration file. Powerful features such as
   radiation and open boundary conditions can be implemented with several
   lines of code. Scripts can be shared for easy reproducible science.
4. Model is continuously tested: unit tests, integration tests, comparison
   with analytic solutions and published scientific results to ensure the
   model is always correct with every commit.
5. GPU benchmarks show that running on a single Nvidia V100 GPU is ~150x
   faster than an Intel Xeon E5-2680. This reduces the lag between
   simulation and analysis, leading to greater research productivity.
   It also enables long integration times of high-resolution simulations.
   Moreover, running on the cloud with GPUs is ~3x more cost-effective.

# Acknowledgements

Funding sources?

We would like to thank Valentin Churavy, Tim Besard, Peter Ahrens, and Alan
Edelman for their Julia support, particularly in developing the GPU capabilities
of ``Oceananigans``.

# References
