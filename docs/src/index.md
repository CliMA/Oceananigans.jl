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

!!! compat "Julia 1.9 or later is required"
    Oceananigans requires Julia 1.9 or later.

!!! info "Tested Julia versions"
    Oceananigans is currently tested on Julia 1.10.

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

## Citing and otherwise spreading the word

If you use Oceananigans for your research, teaching, or fun 🤩, everyone in our community will be grateful
if you credit Oceananigans by name.

The community has published a number of articles describing the development of Oceananigans,
including a recent [preprint submitted to the Journal of Advances in Modeling Earth Systems](https://arxiv.org/abs/2502.14148) that presents an overview of all the things that make Oceananigans unique:

> "High-level, high-resolution ocean modeling at all scales with Oceananigans"
>
> by Gregory L. Wagner, Simone Silvestri, Navid C. Constantinou, Ali Ramadhan, Jean-Michel Campin,
> Chris Hill, Tomas Chor, Jago Strong-Wright, Xin Kai Lee, Francis Poulin, Andre Souza, Keaton J. Burns,
> John Marshall, Raffaele Ferrari
>
> submitted to the Journal of Advances in Modeling Earth Systems, arXiv:2502.14148

Please cite this overview paper if you use Oceananigans in published work.

We've also submitted a number of model development papers. Please cite these if you use
the features they describe! Also, if you have developed a new feature in Oceananigans and describe it in a paper, make sure to open a pull request to add it to this list:

* **Silvestri et al., ["A new WENO-Based momentum advection scheme for simulations of ocean mesoscale turbulence"](https://doi.org/10.1029/2023MS004130).**

  *This paper describes the development of `WENOVectorInvariant()` advection scheme, which can be used as the
  `momentum_advection` scheme for `HydrostaticFreeSurfaceModel`.*

* **Silvestri et al., ["A GPU-based ocean dynamic core for routine mesoscale-resolving climate simulations"](https://doi.org/10.1029/2024MS004465).**

  *This paper describes the optimization of the `HydrostaticFreeSurfaceModel` algorithm, including the implementation
  of a new `SplitExplicitFreeSurface` algorithm for `Distributed` architectures for multiple GPUs. As a result of this work,
  global simulations with O(10 km) grid spacing can be run on 16-20 nodes, achieving 10 simulated years per day (SYPD).*

* **Wagner et al., ["Formulation and calibration of CATKE, a one-equation parameterization for microscale ocean mixing"](https://doi.org/10.1029/2024MS004522).**

  *This paper describes the development of `CATKEVerticalDiffusivity()`, including how it was automatically calibrated to
  a suite of 35 large eddy simulations (also run with Oceananigans). It additionally features solutions from `TKEDissipationVerticalDiffusivity` (also known as "k-epsilon").*

* **Ramadhan et al., ["Oceananigans.jl: Fast and friendly geophysical fluid dynamics on GPUs"](https://par.nsf.gov/servlets/purl/10200806).**

  *This Journal of Open Source Software article describes an early version of Oceananigans' `NonhydrostaticModel`.*

## Papers and preprints using Oceananigans

If you have work using Oceananigans that you would like to have listed here, please open a pull request to add it or let us know!

1. Souza, A. N., Silvestri, S., Deck, K., Bischoff, T., Ferrari, R. and Flierl, G. R. (2025) [Surface to seafloor: A generative AI framework for decoding the ocean interior state](https://doi.org/10.48550/arXiv.2504.15308), _arXiv preprint_, arXiv:2504.15308. DOI: [10.48550/arXiv.2503.12845](https://doi.org/10.48550/arXiv.2504.15308)

1. Johnston, D. R., Shakespeare, C. J., and Constantinou, N. C. (2025) [Evaluating and improving wave and non-wave stress parametrisations for oceanic flows](https://doi.org/10.48550/arXiv.2503.12845), _arXiv preprint_, arXiv:2503.12845 (submitted to the Journal of Physical Oceanography). DOI: [10.48550/arXiv.2503.12845](https://doi.org/10.48550/arXiv.2503.12845)

1. Bhadouriya, A., Gayen, B., Naveira Garabato, A., and Silvano, A. (2025) [Overshooting convection drives winter mixed layer under Antarctic sea ice](https://doi.org/10.21203/rs.3.rs-5932119/v1), preprint (Version 1), available at Research Square. DOI: [10.21203/rs.3.rs-5932119/v1](https://doi.org/10.21203/rs.3.rs-5932119/v1)

1. Whitley V. and Wenegrat, J. O. (2025) [Breaking internal waves on sloping topography: connecting parcel displacements to overturn size, interior-boundary exchanges, and mixing](https://doi.org/10.1175/JPO-D-24-0052.1), _Journal of Physical Oceanography_. In press. DOI: [10.1175/JPO-D-24-0052.1](https://doi.org/10.1175/JPO-D-24-0052.1)

1. Silvestri, S., Wagner, G. L., Constantinou, N. C., Hill, C., Campin, J.-M., Souza, A., Bishnu, S., Churavy, V., Marshall, J., and Ferrari, R. (2025) [A GPU-based ocean dynamical core for routine mesoscale-resolving climate simulations](https://doi.org/10.1029/2024MS004465), _Journal of Advances in Modeling Earth Systems_, **16(7)**, e2024MS004465. DOI: [10.1029/2024MS004465](https://doi.org/10.1029/2024MS004465)

1. Wagner, G. L., Hillier, A., Constantinou, N. C., Silvestri, S., Souza, A., Burns, K., Hill, C., Campin, J.-M., Marshall, J., and Ferrari, R. (2025). [Formulation and calibration of CATKE, a one-equation parameterization for microscale ocean mixing](https://doi.org/10.1029/2024MS004522), _Journal of Advances in Modeling Earth Systems_, **16(7)**, e2024MS004522. DOI: [10.1029/2024MS004522](https://doi.org/10.1029/2024MS004522)

1. Fan, X., Fox-Kemper, B., Suzuki, N., Li, Q., Marchesiello, P., Sullivan, P. P., and Hall, P. S. (2024) [Comparison of the Coastal and Regional Ocean COmmunity model (CROCO) and NCAR-LES in non-hydrostatic simulations](https://doi.org/10.5194/gmd-17-4095-2024), _Geoscientific Model Development_, **17**, 4095-4113. DOI: [10.5194/gmd-17-4095-2024](https://doi.org/10.5194/gmd-17-4095-2024)

1. Abbott, K. and Mahadevan, A. (2024). [Why is the monsoon coastal upwelling signal subdued in the Bay of Bengal?](https://doi.org/10.1029/2024JC022023), _Journal of Geophysical Research: Oceans_, **129**, e2024JC022023. DOI: [10.1029/2024JC022023](https://doi.org/10.1029/2024JC022023)

1. Bisits, J. I., Zika, J. D., and Evans, D. G. (2024) [Does cabbeling shape the thermohaline structure of high-latitude oceans?](https://doi.org/10.1175/JPO-D-24-0061.1), _Journal of Physical Oceanography_, in press. DOI: [10.1175/JPO-D-24-0061.1](https://doi.org/10.1175/JPO-D-24-0061.1)

1. Chor, T. and Wenegrat, J. (2024). [The turbulent dynamics of anticyclonic submesoscale headland wakes](https://doi.org/10.31223/X5570C), _Earth arXiv_, DOI: [10.31223/X5570C](https://doi.org/10.31223/X5570C)

1. Allred, T., Li, X., Wiersdorf, A., Greenman, B., and Gopalakrishnan, G. (2024). [FlowFPX: Nimble tools for debugging floating-point exceptions](https://doi.org/10.48550/arXiv.2403.15632), _arXiv_, arXiv:2403.15632. DOI: [10.48550/arXiv.2403.15632](https://doi.org/10.48550/arXiv.2403.15632)

1. Strong-Wright J. and Taylor, J. R. (2024) [A model of tidal flow and tracer release in a giant kelp forest](https://doi.org/10.1017/flo.2024.13), _Flow_, **4**, E21. DOI: [10.1017/flo.2024.13](https://doi.org/10.1017/flo.2024.13)

1. Silvestri, S., Wagner, G. L., Campin, J.-M., Constantinou, N. C., Hill, C., Souza, A., and Ferrari, R. (2024). [A new WENO-based momentum advection scheme for simulations of ocean mesoscale turbulence](https://doi.org/10.1029/2023MS004130), _Journal of Advances in Modeling Earth Systems_, **16(7)**, e2023MS004130. DOI: [10.1029/2023MS004130](https://doi.org/10.1029/2023MS004130)

1. Chen S., Strong-Wright J., and Taylor, J. R. (2024) [Modeling carbon dioxide removal via sinking of particulate organic carbon from macroalgae cultivation](https://doi.org/10.3389/fmars.2024.1359614), _Frontiers in Marine Science_, **11**, 1359614. DOI: [10.3389/fmars.2024.1359614](https://doi.org/10.3389/fmars.2024.1359614)

1. Gupta, M., Gürcan, E., and Thompson, A. F. (2024). [Eddy-induced dispersion of sea ice floes at the marginal ice zone](https://doi.org/10.1029/2023GL105656), _Geophysical Research Letters_, **51**, e2023GL105656. DOI: [10.1029/2023GL105656](https://doi.org/10.1029/2023GL105656)

1. Wagner, G. L., Pizzo, N. E., Lenain, L., and Veron, F. (2023) [Transition to turbulence in wind-drift layers](https://doi.org/10.1017/jfm.2023.920), _Journal of Fluid Mechanics_, **976**, A8. DOI: [10.1017/jfm.2023.920](https://doi.org/10.1017/jfm.2023.920)

1. Jiménez-Urias, M. A. and Haine T. W. N. (2023) [On the non-self-adjoint and multiscale character of passive scalar mixing under laminar advection](https://doi.org/10.1017/jfm.2023.748), _Journal of Fluid Mechanics_, **973**, A44. DOI: [10.1017/jfm.2023.748](https://doi.org/10.1017/jfm.2023.748)

1. Strong-Wright, J, Chen, S., Constantinou, N. C., Silvestri, S., Wagner, G. L., and Taylor, J. R. (2023). [OceanBioME.jl: A flexible environment for modelling the coupled interactions between ocean biogeochemistry and physics](https://doi.org/10.21105/joss.05669), _Journal of Open Source Software_, **90(8)**, 5669. DOI: [10.21105/joss.05669](https://doi.org/10.21105/joss.05669)

1. Ramadhan, A., Marshall, J. C., Souza, A. N., Lee, X. K., Piterbarg, U., Hillier, A., Wagner, G. L., Rackauckas, C., Hill, C., Campin, J.-M., and Ferrari, R. (2022). [Capturing missing physics in climate model parameterizations using neural differential equations](https://doi.org/10.1002/essoar.10512533.1), _ESS Open Archive_. DOI: [10.1002/essoar.10512533.1](https://doi.org/10.1002/essoar.10512533.1)

1. Gupta, M. and Thompson, A. F. (2022). [Regimes of sea-ice floe melt: Ice-ocean coupling at the submesoscales](https://doi.org/10.1029/2022JC018894), _Journal of Geophysical Research: Oceans_, **127**, e2022JC018894. DOI: [10.1029/2022JC018894](https://doi.org/10.1029/2022JC018894)

1. Simoes-Sousa, I. T., Tandon, A., Pereira, F., Lazaneo, C. Z., and Mahadevan, A. (2022). [Mixed layer eddies supply nutrients to enhance the spring phytoplankton bloom](https://doi.org/10.3389/fmars.2022.825027), _Frontiers in Marine Sciences_, **9**, 825027. DOI: [10.3389/fmars.2022.825027](https://doi.org/10.3389/fmars.2022.825027)

1. Chor, T., Wenegrat, J. O., and Taylor, J. (2022). [Insights into the mixing efficiency of submesoscale Centrifugal-Symmetric instabilities.](https://doi.org/10.1175/JPO-D-21-0259.1), _Journal of Physical Oceanography_, **52(10)**, 2273-2287. DOI: [10.1175/JPO-D-21-0259.1](https://doi.org/10.1175/JPO-D-21-0259.1)

1. Bire, S., Kang, W., Ramadhan, A., Campin, J.-M., and Marshall, J. (2022). [Exploring ocean circulation on icy moons heated from below.](https://doi.org/10.1029/2021JE007025), _Journal of Geophysical Research: Planets_, **127**, e2021JE007025. DOI: [10.1029/2021JE007025](https://doi.org/10.1029/2021JE007025)

1. Rackauckas, C., Ma, Y., Martensen, J., Warner, C., Zubov, K., Supekar, R., Skinner, D., Ramadhan, A., and Edelman, A. (2021) [Universal differential equations for scientific machine learning](https://doi.org/10.48550/arXiv.2001.04385), _arXiv_, arXiv.2001.04385. DOI: [10.48550/arXiv.2001.04385](https://doi.org/10.48550/arXiv.2001.04385)

1. Coakley, S., Miles, T. N., Glenn, S., and Lim, H. S. (2021). [Observation-Large eddy simulation comparison of ocean mixing under Typhoon Soulik (2018)](https://doi.org/10.23919/OCEANS44145.2021.9705670), _OCEANS 2021: San Diego – Porto, 2021_, pp. 1-7. DOI: [10.23919/OCEANS44145.2021.9705670](https://doi.org/10.23919/OCEANS44145.2021.9705670)

1. Arnscheidt, C. W., Marshall, J., Dutrieux, P., Rye, C. D., and Ramadhan, A. (2021). [On the settling depth of meltwater escaping from beneath Antarctic ice shelves](https://doi.org/10.1175/JPO-D-20-0286.1), _Journal of Physical Oceanography_, **51(7)**, 2257–2270. DOI: [10.1175/JPO-D-20-0178.1](https://doi.org/10.1175/JPO-D-20-0286.1)

1. Wagner, G. L., Chini, G. P., Ramadhan, A., Gallet, B., and Ferrari, R. (2021). [Near-inertial waves and turbulence driven by the growth of swell](https://doi.org/10.1175/JPO-D-20-0178.1), _Journal of Physical Oceanography_, **51(5)**, 1337-1351. DOI: [10.1175/JPO-D-20-0178.1](https://doi.org/10.1175/JPO-D-20-0178.1)

1. Buffett, B. A. (2021). [Conditions for turbulent Ekman layers in precessionally driven flow](https://doi.org/10.1093/gji/ggab088), _Geophysical Journal International_, **226(1)**, 56–65. DOI: [10.1093/gji/ggab088](https://doi.org/10.1093/gji/ggab088)

1. Bhamidipati, N., Souza, A.N., and Flierl, G.R. (2020). [Turbulent mixing of a passive scalar in the ocean mixed layer](https://doi.org/10.1016/j.ocemod.2020.101615), _Ocean Modelling_, **149**, 101615. DOI: [10.1016/j.ocemod.2020.101615](https://doi.org/10.1016/j.ocemod.2020.101615)

1. Souza, A. N., Wagner, G. L., Ramadhan, A., Allen, B., Churavy, V., Schloss, J., Campin, J. M., Hill, C., Edelman, A., Marshall, J., Flierl, G., and Ferrari, R. (2020). [Uncertainty quantification of ocean parameterizations: Application to the K‐Profile‐Parameterization for penetrative convection](https://doi.org/10.1029/2020MS002108), _Journal of Advances in Modeling Earth Systems_, **12**, e2020MS002108. DOI: [10.1029/2020MS002108](https://doi.org/10.1029/2020MS002108)
