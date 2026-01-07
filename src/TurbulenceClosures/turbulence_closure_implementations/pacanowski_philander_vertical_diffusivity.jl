using Oceananigans.Architectures: architecture
using Oceananigans.BuoyancyFormulations: ∂z_b
using Oceananigans.Operators

"""
    PacanowskiPhilanderVerticalDiffusivity{TD, FT} <: AbstractScalarDiffusivity{TD, VerticalFormulation, 1}

A turbulence closure that parameterizes vertical viscosity and diffusivity as functions
of the Richardson number following Pacanowski and Philander (1981). The formulation is:

```math
\\nu = \\nu_0 + \\frac{\\nu_1}{(1 + c \\, Ri)^n}
```

```math
\\kappa = \\kappa_0 + \\frac{\\nu_1}{(1 + c \\, Ri)^{n+1}}
```

where ``Ri`` is the gradient Richardson number, ``\\nu_0`` and ``\\kappa_0`` are background
viscosity and diffusivity, ``\\nu_1`` is the shear-driven viscosity coefficient, ``c`` is a
scaling parameter for the Richardson number, and ``n`` is the exponent.

References
==========

Pacanowski, R. C., & Philander, S. G. H. (1981). Parameterization of vertical mixing in
numerical models of tropical oceans. Journal of Physical Oceanography, 11(11), 1443-1451.
"""
struct PacanowskiPhilanderVerticalDiffusivity{TD, FT} <: AbstractScalarDiffusivity{TD, VerticalFormulation, 1}
    ν₀ :: FT  # Background viscosity
    ν₁ :: FT  # Shear-driven viscosity coefficient
    κ₀ :: FT  # Background diffusivity
    c  :: FT  # Richardson number scaling coefficient
    n  :: FT  # Exponent for viscosity
    maximum_diffusivity :: FT
    maximum_viscosity :: FT
end

function PacanowskiPhilanderVerticalDiffusivity{TD}(ν₀::FT, ν₁::FT, κ₀::FT, c::FT, n::FT,
                                                    maximum_diffusivity::FT,
                                                    maximum_viscosity::FT) where {TD, FT}
    return PacanowskiPhilanderVerticalDiffusivity{TD, FT}(ν₀, ν₁, κ₀, c, n,
                                                          maximum_diffusivity,
                                                          maximum_viscosity)
end

"""
    PacanowskiPhilanderVerticalDiffusivity([time_discretization = VerticallyImplicitTimeDiscretization(),
                                            FT = Float64;]
                                            ν₀ = 1e-4,
                                            ν₁ = 1e-2,
                                            κ₀ = 1e-5,
                                            c  = 5.0,
                                            n  = 2.0,
                                            maximum_diffusivity = Inf,
                                            maximum_viscosity = Inf)

Return a `PacanowskiPhilanderVerticalDiffusivity` turbulence closure that parameterizes
vertical mixing as a function of the gradient Richardson number following
Pacanowski and Philander (1981).

The viscosity and diffusivity are computed as

```math
\\nu = \\nu_0 + \\frac{\\nu_1}{(1 + c \\, Ri)^n}
```

```math
\\kappa = \\kappa_0 + \\frac{\\nu_1}{(1 + c \\, Ri)^{n+1}}
```

where ``Ri = N^2 / S^2`` is the gradient Richardson number, ``N^2 = \\partial_z b`` is
the buoyancy frequency squared, and ``S^2 = (\\partial_z u)^2 + (\\partial_z v)^2``
is the vertical shear squared.

Arguments
=========

* `time_discretization`: Either `ExplicitTimeDiscretization()` or `VerticallyImplicitTimeDiscretization()`.
                         Default is `VerticallyImplicitTimeDiscretization()`.

* `FT`: Float type; default `Float64`.

Keyword Arguments
=================

* `ν₀`: Background viscosity (m² s⁻¹). Default: `1e-4`.

* `ν₁`: Shear-driven viscosity coefficient (m² s⁻¹). Default: `1e-2`.

* `κ₀`: Background diffusivity (m² s⁻¹). Default: `1e-5`.

* `c`: Richardson number scaling coefficient (dimensionless). Default: `5.0`.

* `n`: Exponent for viscosity (dimensionless). Default: `2.0`.
       Note: diffusivity uses exponent `n + 1`.

* `maximum_diffusivity`: A limiting maximum diffusivity (m² s⁻¹). Default: `Inf`.

* `maximum_viscosity`: A limiting maximum viscosity (m² s⁻¹). Default: `Inf`.

Example
=======

```jldoctest
julia> using Oceananigans

julia> closure = PacanowskiPhilanderVerticalDiffusivity()
PacanowskiPhilanderVerticalDiffusivity{VerticallyImplicitTimeDiscretization}:
├── ν₀: 0.0001
├── ν₁: 0.01
├── κ₀: 1.0e-5
├── c: 5.0
└── n: 2.0
```

References
==========

Pacanowski, R. C., & Philander, S. G. H. (1981). Parameterization of vertical mixing in
numerical models of tropical oceans. Journal of Physical Oceanography, 11(11), 1443-1451.
"""
function PacanowskiPhilanderVerticalDiffusivity(time_discretization = VerticallyImplicitTimeDiscretization(),
                                                FT = Oceananigans.defaults.FloatType;
                                                ν₀ = 1e-4,
                                                ν₁ = 1e-2,
                                                κ₀ = 1e-5,
                                                c  = 5.0,
                                                n  = 2.0,
                                                maximum_diffusivity = Inf,
                                                maximum_viscosity = Inf)

    TD = typeof(time_discretization)

    return PacanowskiPhilanderVerticalDiffusivity{TD}(convert(FT, ν₀),
                                                      convert(FT, ν₁),
                                                      convert(FT, κ₀),
                                                      convert(FT, c),
                                                      convert(FT, n),
                                                      convert(FT, maximum_diffusivity),
                                                      convert(FT, maximum_viscosity))
end

PacanowskiPhilanderVerticalDiffusivity(FT::DataType; kw...) =
    PacanowskiPhilanderVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kw...)

#####
##### Type aliases and diffusivity field utilities
#####

const PPVD = PacanowskiPhilanderVerticalDiffusivity
const PPVDArray = AbstractArray{<:PPVD}
const FlavorOfPPVD = Union{PPVD, PPVDArray}

const c = Center()
const f = Face()

@inline viscosity_location(::FlavorOfPPVD)   = (c, c, f)
@inline diffusivity_location(::FlavorOfPPVD) = (c, c, f)

@inline viscosity(::FlavorOfPPVD, diffusivities) = diffusivities.νz
@inline diffusivity(::FlavorOfPPVD, diffusivities, id) = diffusivities.κz

Utils.with_tracers(tracers, closure::FlavorOfPPVD) = closure

function build_closure_fields(grid, clock, tracer_names, bcs, closure::FlavorOfPPVD)
    κz = Field{Center, Center, Face}(grid)
    νz = Field{Center, Center, Face}(grid)
    return (; κz, νz)
end

#####
##### Richardson number and diffusivity computation
#####

# Note: shear_squaredᶜᶜᶠ and ϕ² are shared with RiBasedVerticalDiffusivity
# (defined in ri_based_vertical_diffusivity.jl which is included first)

@inline function Riᶜᶜᶠ_pp(i, j, k, grid, velocities, buoyancy, tracers)
    S² = shear_squaredᶜᶜᶠ(i, j, k, grid, velocities)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)

    # Clip N² at zero (ignore unstable stratification for Ri)
    # Avoid division by zero by returning zero Ri when S² is negligible
    S²_min = eps(eltype(grid))
    Ri = max(zero(grid), N²) / max(S², S²_min)

    return Ri
end

function compute_diffusivities!(diffusivities, closure::FlavorOfPPVD, model; parameters = :xyz)
    arch = model.architecture
    grid = model.grid
    tracers = buoyancy_tracers(model)
    buoyancy = buoyancy_force(model)
    velocities = model.velocities

    launch!(arch, grid, parameters,
            compute_pacanowski_philander_diffusivities!,
            diffusivities,
            grid,
            closure,
            velocities,
            tracers,
            buoyancy)

    return nothing
end

@kernel function compute_pacanowski_philander_diffusivities!(diffusivities, grid, closure,
                                                             velocities, tracers, buoyancy)
    i, j, k = @index(Global, NTuple)

    # Support "ensembles" of closures (closure arrays)
    closure_ij = getclosure(i, j, closure)

    Ri = Riᶜᶜᶠ_pp(i, j, k, grid, velocities, buoyancy, tracers)

    # Extract parameters
    ν₀ = closure_ij.ν₀
    ν₁ = closure_ij.ν₁
    κ₀ = closure_ij.κ₀
    cc = closure_ij.c   # use cc to avoid shadowing the const c
    n  = closure_ij.n

    # Pacanowski-Philander (1981) formulas
    # ν = ν₀ + ν₁ / (1 + c * Ri)^n
    # κ = κ₀ + ν₁ / (1 + c * Ri)^(n+1)
    denominator = 1 + cc * Ri
    νz = ν₀ + ν₁ / denominator^n
    κz = κ₀ + ν₁ / denominator^(n + 1)

    # Apply maximum limits
    νz = min(νz, closure_ij.maximum_viscosity)
    κz = min(κz, closure_ij.maximum_diffusivity)

    @inbounds diffusivities.νz[i, j, k] = νz
    @inbounds diffusivities.κz[i, j, k] = κz
end

#####
##### Show
#####

Base.summary(closure::PPVD{TD}) where TD = string("PacanowskiPhilanderVerticalDiffusivity{$TD}")

function Base.show(io::IO, closure::PPVD)
    print(io, summary(closure), ":\n")
    print(io, "├── ν₀: ", prettysummary(closure.ν₀), '\n')
    print(io, "├── ν₁: ", prettysummary(closure.ν₁), '\n')
    print(io, "├── κ₀: ", prettysummary(closure.κ₀), '\n')
    print(io, "├── c: ", prettysummary(closure.c), '\n')
    print(io, "└── n: ", prettysummary(closure.n))
end

