# Oceananigans.jl

*🌊 Fast and friendly fluid dynamics on CPUs and GPUs.*

Oceananigans is fast, friendly, and flexible software for finite volume simulations the nonhydrostatic
and hydrostatic Boussinesq equations on CPUs and GPUs. Aside from its ability to run on GPUs (wow, fast!)
Oceananigans receives its powers from a flexible and intuitive user interface that
makes simple numerical experiments easy, and complex, creative numerical experiments possible.

Oceananigans is written in Julia by the [Climate Modeling Alliance](https://clima.caltech.edu)
and heroic external collaborators.

## Quick install

Oceananigans is a [registered Julia package](https://julialang.org/packages/), so the installation instructions are:

1. [Download Julia 1.6](https://julialang.org/downloads/);

2. Start up Julia and type

```julia
julia> using Pkg

julia> Pkg.add("Oceananigans")
```

If you're [new to Julia](https://docs.julialang.org/en/v1/manual/getting-started/) and using [Julia's wonderful and featureful package manager `Pkg`](https://docs.julialang.org/en/v1/stdlib/Pkg/), or just like reading documentation,
the Oceananigans wiki provides [more detailed installation instructions](https://github.com/CliMA/Oceananigans.jl/wiki/Installation-and-getting-started-with-Oceananigans).

Note: that many Oceananigans experiments will run on Julia 1.7, it's just that our tests don't cover Julia 1.7 _yet_.

## The Oceananigans "knowledge base"

Oceananigans knowledge runs _deep_ and includes:

* This documentation, covering Oceananigans syntax, usage, code structure, and a description of
* [Discussions on the Oceananigans github](https://github.com/CliMA/Oceananigans.jl/discussions), covering topics like
    * ["computational science"](https://github.com/CliMA/Oceananigans.jl/discussions/categories/computational-science) (how to science and set up numerical simulations in Oceananigans) and
    * ["experimental features"](https://github.com/CliMA/Oceananigans.jl/discussions?discussions_q=experimental+features) (for those who like to live a little).
    
    If you've got a question or something to talk about, don't hestitate to [start a new discussion](https://github.com/CliMA/Oceananigans.jl/discussions/new?)!
* The [Oceananigans wiki](https://github.com/CliMA/Oceananigans.jl/wiki), which contains practical tips for [getting started with Julia](https://github.com/CliMA/Oceananigans.jl/wiki/Installation-and-getting-started-with-Oceananigans), [accessing and using GPUs](https://github.com/CliMA/Oceananigans.jl/wiki/Oceananigans-on-GPUs), and best practices for using Oceananigans.
* [Issues](https://github.com/CliMA/Oceananigans.jl/issues) and [pull requests](https://github.com/CliMA/Oceananigans.jl/pulls) also contain lots of information about problems we've found, solutions we're trying to implement, and dreams we're dreaming to make tomorrow better 🌈.

## Getting in touch

Whether you need help getting started with Oceananigans, found a bug, want Oceananigans to be more awesome, or just want to chat about computational oceanography, you've got a few options for getting in touch:

* [Start a discussion](https://github.com/CliMA/Oceananigans.jl/discussions). This is great for general questions about numerics, science, experimental or under-documented features, and for getting help setting up a neat new numerical experiment.
* [Open an issue](https://github.com/CliMA/Oceananigans.jl/issues). Issues are best if you think the Oceananigans source code needs attention: a bug, a sign error (😱), an important missing feature, or a typo in this documentation 👀.
* Sign up for the [Julia Slack](https://julialang.org/slack/) and join the `#oceananigans` channel. We love to chat.

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
