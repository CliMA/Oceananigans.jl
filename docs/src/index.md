# Oceananigans.jl

*🌊 Fast and friendly fluid dynamics on CPUs and GPUs.*

Oceananigans is a fast, friendly, flexible software package for finite volume simulations of the nonhydrostatic
and hydrostatic Boussinesq equations on CPUs and GPUs.
It runs on GPUs (wow, fast!), though we believe Oceananigans makes the biggest waves
with its ultra-flexible user interface that makes simple simulations easy, and complex, creative simulations possible.

Oceananigans is written in Julia by the [Climate Modeling Alliance](https://clima.caltech.edu)
and heroic external collaborators.

## Quick install

Oceananigans is a [registered Julia package](https://julialang.org/packages/). So to install it,

1. [Download Julia](https://julialang.org/downloads/).

2. Launch Julia and type

```julia
julia> using Pkg

julia> Pkg.add("Oceananigans")
```

!!! compat "Julia 1.6 or newer"
    The latest version of Oceananigans strongly suggests _at least_ Julia 1.8 or later to run.
    While most scripts will run on Julia 1.6 or 1.7, Oceananigans is _only_ tested on Julia 1.8.

If you're [new to Julia](https://docs.julialang.org/en/v1/manual/getting-started/) and its [wonderful `Pkg` manager](https://docs.julialang.org/en/v1/stdlib/Pkg/), the [Oceananigans wiki](https://github.com/CliMA/Oceananigans.jl/wiki) provides [more detailed installation instructions](https://github.com/CliMA/Oceananigans.jl/wiki/Installation-and-getting-started-with-Oceananigans).

## The Oceananigans "knowledge base"

It's _deep_ and includes:

* This documentation, which provides
    * example Oceananigans scripts,
    * tutorials that describe key Oceananigans objects and functions,
    * explanations of Oceananigans finite-volume-based numerical methods,
    * details of the dynamical equations solved by Oceananigans models, and
    * a library documenting all user-facing Oceananigans objects and functions.
* [Discussions on the Oceananigans github](https://github.com/CliMA/Oceananigans.jl/discussions), covering topics like
    * ["Computational science"](https://github.com/CliMA/Oceananigans.jl/discussions/categories/computational-science), or how to science and set up numerical simulations in Oceananigans, and
    * ["Experimental features"](https://github.com/CliMA/Oceananigans.jl/discussions?discussions_q=experimental+features), which covers new and sparsely-documented features for those who like to live dangerously.
  
    If you've got a question or something to talk about, don't hesitate to [start a new discussion](https://github.com/CliMA/Oceananigans.jl/discussions/new?)!
* The [Oceananigans wiki](https://github.com/CliMA/Oceananigans.jl/wiki), which contains practical tips for [getting started with Julia](https://github.com/CliMA/Oceananigans.jl/wiki/Installation-and-getting-started-with-Oceananigans), [accessing and using GPUs](https://github.com/CliMA/Oceananigans.jl/wiki/Oceananigans-on-GPUs), and [productive workflows when using Oceananigans](https://github.com/CliMA/Oceananigans.jl/wiki/Productive-Oceananigans-workflows-and-Julia-environments).
* [Issues](https://github.com/CliMA/Oceananigans.jl/issues) and [pull requests](https://github.com/CliMA/Oceananigans.jl/pulls) also contain lots of information about problems we've found, solutions we're trying to implement, and dreams we're dreaming to make tomorrow better 🌈.

## Getting in touch

Whether you need help getting started with Oceananigans, found a bug, want Oceananigans to be more awesome, or just want to chat about computational oceanography, you've got a few options for getting in touch:

* [Start a discussion](https://github.com/CliMA/Oceananigans.jl/discussions). This is great for general questions about numerics, science, experimental or under-documented features, and for getting help setting up a neat new numerical experiment.
* [Open an issue](https://github.com/CliMA/Oceananigans.jl/issues). Issues are best if you think the Oceananigans source code needs attention: a bug, a sign error (😱), an important missing feature, or a typo in this documentation 👀.
* Sign up for the [Julia Slack](https://julialang.org/slack/) and join the `#oceananigans` channel because we love to chat.

## Citing

If you use Oceananigans as part of your research, teaching, or other activities, we would be grateful if you could
cite our work and mention Oceananigans by name.

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

## Papers and preprints using Oceananigans

If you have work using Oceananigans that you would like to have listed here, please open a pull request to add it or let us know!

1. Ramadhan, A., Marshall, J. C., Souza, A. N., Lee, X. K., Piterbarg, U., Hillier, A., Wagner, G. L., Rackauckas, C., Hill, C., Campin, J.-M., and Ferrari, R. (2022). [Capturing missing physics in climate model parameterizations using neural differential equations](https://doi.org/10.1002/essoar.10512533.1) _ESS Open Archive_. DOI: [10.1002/essoar.10512533.1](https://doi.org/10.1002/essoar.10512533.1)

1. Gupta, M., & Thompson, A. F. (2022). [Regimes of sea-ice floe melt: Ice-ocean coupling at the submesoscales](https://doi.org/10.1029/2022JC018894) _Journal of Geophysical Research: Oceans_, **127**, e2022JC018894. DOI: [10.1029/2022JC018894](https://doi.org/10.1029/2022JC018894)

1. Simoes-Sousa, I. T., Tandon, A., Pereira, F., Lazaneo, C. Z., and Mahadevan, A. (2022). [Mixed layer eddies supply nutrients to enhance the spring phytoplankton bloom](https://doi.org/10.3389/fmars.2022.825027) _Frontiers in Marine Sciences_, **9**, 825027. DOI: [10.3389/fmars.2022.825027](https://doi.org/10.3389/fmars.2022.825027)

1. Chor, T., Wenegrat, J. O., and Taylor, J. (2022). [Insights into the mixing efficiency of submesoscale Centrifugal-Symmetric instabilities.](https://doi.org/10.1175/JPO-D-21-0259.1) _Journal of Physical Oceanography_, **52(10)**, 2273-2287. DOI: [10.1175/JPO-D-21-0259.1](https://doi.org/10.1175/JPO-D-21-0259.1)

1. Bire, S., Kang, W., Ramadhan, A., Campin, J.-M., and Marshall, J. (2022). [Exploring ocean circulation on icy moons heated from below.](https://doi.org/10.1029/2021JE007025) _Journal of Geophysical Research: Planets_, **127**, e2021JE007025. DOI: [10.1029/2021JE007025](https://doi.org/10.1029/2021JE007025)

1. Coakley, S., Miles, T. N., Glenn, S., and Lim, H. S. (2021). [Observation-Large eddy simulation comparison of ocean mixing under Typhoon Soulik (2018)](https://doi.org/10.23919/OCEANS44145.2021.9705670), _OCEANS 2021: San Diego – Porto, 2021_, pp. 1-7, DOI: [10.23919/OCEANS44145.2021.9705670](https://doi.org/10.23919/OCEANS44145.2021.9705670)

1. Arnscheidt, C. W., Marshall, J., Dutrieux, P., Rye, C. D., and Ramadhan, A. (2021). [On the settling depth of meltwater escaping from beneath Antarctic ice shelves](https://doi.org/10.1175/JPO-D-20-0286.1), _Journal of Physical Oceanography_, **51(7)**, 2257–2270. DOI: [10.1175/JPO-D-20-0178.1](https://doi.org/10.1175/JPO-D-20-0286.1)

1. Wagner, G. L., Chini, G. P., Ramadhan, A., Gallet, B., and Ferrari, R. (2021). [Near-inertial waves and turbulence driven by the growth of swell](https://doi.org/10.1175/JPO-D-20-0178.1), _Journal of Physical Oceanography_, **51(5)**, 1337-1351. DOI: [10.1175/JPO-D-20-0178.1](https://doi.org/10.1175/JPO-D-20-0178.1)

1. Buffett, B. A. (2021). [Conditions for turbulent Ekman layers in precessionally driven flow](https://doi.org/10.1093/gji/ggab088), _Geophysical Journal International_, **226(1)**, 56–65. DOI: [10.1093/gji/ggab088](https://doi.org/10.1093/gji/ggab088)

1. Bhamidipati, N., Souza, A.N., and Flierl, G.R. (2020). [Turbulent mixing of a passive scalar in the ocean mixed layer](https://doi.org/10.1016/j.ocemod.2020.101615). _Ocean Modelling_, **149**, 101615. DOI: [10.1016/j.ocemod.2020.101615](https://doi.org/10.1016/j.ocemod.2020.101615)

1. Souza, A. N., Wagner, G. L., Ramadhan, A., Allen, B., Churavy, V., Schloss, J., Campin, J. M., Hill, C., Edelman, A., Marshall, J., Flierl, G., and Ferrari, R. (2020). [Uncertainty quantification of ocean parameterizations: Application to the K‐Profile‐Parameterization for penetrative convection](https://doi.org/10.1029/2020MS002108). _Journal of Advances in Modeling Earth Systems_, **12**, e2020MS002108. DOI: [10.1029/2020MS002108](https://doi.org/10.1029/2020MS002108)
