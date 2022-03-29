# Oceananigans.jl

*🌊 Fast and friendly fluid dynamics on CPUs and GPUs.*

Oceananigans.jl is a fast and friendly fluid flow solver written in Julia that can be run in 1-3 dimensions on CPUs
and GPUs. It can simulate the incompressible Boussinesq equations, the shallow water equations, or the hydrostatic
Boussinesq equations with a free surface. Oceananigans.jl comes with user-friendly features for simulating rotating
stratified fluids including user-defined boundary conditions and forcing functions, arbitrary tracers, large eddy
simulation turbulence closures, high-order advection schemes, immersed boundaries, Lagrangian particle tracking, and
more!

We strive for a user interface that makes Oceananigans.jl as friendly and intuitive to use as possible,
allowing users to focus on the science. Internally, we have attempted to write the underlying algorithm
so that the code runs as fast as possible for the configuration chosen by the user --- from simple
two-dimensional setups to complex three-dimensional simulations --- and so that as much code
as possible is shared between the different architectures, models, and grids.

## Getting help

If you are interested in using Oceananigans.jl or are trying to figure out how to use it, please feel free to ask us
questions and get in touch! If you're trying to set up a model then check out the examples and model setup
documentation. Please feel free to [start a discussion](https://github.com/CliMA/Oceananigans.jl/discussions)
if you have any questions, comments, suggestions, etc! There is also an #oceananigans channel on the
[Julia Slack](https://julialang.org/slack/).

## Citing

If you use Oceananigans.jl as part of your research, teaching, or other activities, we would be grateful if you could
cite our work and mention Oceananigans.jl by name.

```bibtex
@article{OceananigansJOSS,
  doi = {10.21105/joss.02018},
  url = {https://doi.org/10.21105/joss.02018},
  year = {2020},
  publisher = {The Open Journal},
  volume = {5},
  number = {53},
  pages = {2018},
  author = {Ali Ramadhan and Gregory LeClaire Wagner and Chris Hill and Jean-Michel Campin and Valentin Churavy and Tim Besard and Andre Souza and Alan Edelman and Raffaele Ferrari and John Marshall},
  title = {Oceananigans.jl: Fast and friendly geophysical fluid dynamics on GPUs},
  journal = {Journal of Open Source Software}
}
```

## Papers and preprints using Oceananigans.jl

If you have work using Oceananigans.jl that you would like to have listed here, please open a pull request to add it or let us know!

1. Bire, S., Kang, W., Ramadhan, A., Campin, J.-M., and Marshall, J. (2022). [Exploring ocean circulation on icy moons heated from below.](https://doi.org/10.1029/2021JE007025) _Journal of Geophysical Research: Planets_, **127**, e2021JE007025. DOI: [10.1029/2021JE007025](https://doi.org/10.1029/2021JE007025)

1. Arnscheidt, C. W., Marshall, J., Dutrieux, P., Rye, C. D., and Ramadhan, A. (2021). [On the settling depth of meltwater escaping from beneath Antarctic ice shelves](https://doi.org/10.1175/JPO-D-20-0286.1), _Journal of Physical Oceanography_, **51(7)**, 2257–2270. DOI: [10.1175/JPO-D-20-0178.1](https://doi.org/10.1175/JPO-D-20-0286.1)

1. Wagner, G. L., Chini, G. P., Ramadhan, A., Gallet, B., and Ferrari, R. (2021). [Near-inertial waves and turbulence driven by the growth of swell](https://doi.org/10.1175/JPO-D-20-0178.1), _Journal of Physical Oceanography_, **51(5)**, 1337-1351. DOI: [10.1175/JPO-D-20-0178.1](https://doi.org/10.1175/JPO-D-20-0178.1)

1. Buffett, B. A. (2021). [Conditions for turbulent Ekman layers in precessionally driven flow](https://doi.org/10.1093/gji/ggab088), _Geophysical Journal International_, **226(1)**, 56–65. DOI: [10.1093/gji/ggab088](https://doi.org/10.1093/gji/ggab088)

1. Bhamidipati, N., Souza, A.N., and Flierl, G.R. (2020). [Turbulent mixing of a passive scalar in the ocean mixed layer](https://doi.org/10.1016/j.ocemod.2020.101615). _Ocean Modelling_, **149**, 101615. DOI: [10.1016/j.ocemod.2020.101615](https://doi.org/10.1016/j.ocemod.2020.101615)

1. Souza, A. N., Wagner, G. L., Ramadhan, A., Allen, B., Churavy, V., Schloss, J., Campin, J. M., Hill, C., Edelman, A., Marshall, J., Flierl, G., and Ferrari, R. (2020). [Uncertainty quantification of ocean parameterizations: Application to the K‐Profile‐Parameterization for penetrative convection](https://doi.org/10.1029/2020MS002108). _Journal of Advances in Modeling Earth Systems_, **12**, e2020MS002108. DOI: [10.1029/2020MS002108](https://doi.org/10.1029/2020MS002108)
