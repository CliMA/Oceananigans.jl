module MEWSVerticalDiffusivities

using Adapt
using KernelAbstractions: @kernel, @index

using Oceananigans.Utils
using Oceananigans.Fields
using Oceananigans.Operators

using Oceananigans.Architectures: device, device_event
using Oceananigans.Fields: ZeroField
using Oceananigans.BoundaryConditions: FluxBoundaryCondition, FieldBoundaryConditions
using Oceananigans.BuoyancyModels: ∂x_b, ∂y_b, ∂z_b

import Oceananigans.Grids: required_halo_size

using Oceananigans.Utils: prettysummary
using Oceananigans.Coriolis: fᶠᶠᵃ

using Oceananigans.TurbulenceClosures:
    wall_vertical_distanceᶜᶜᶠ,
    opposite_wall_vertical_distanceᶜᶜᶠ,
    getclosure,
    AbstractScalarDiffusivity,
    VerticallyImplicitTimeDiscretization,
    VerticalFormulation

import Oceananigans.Utils: with_tracers

import Oceananigans.TurbulenceClosures:
    shear_production,
    buoyancy_flux,
    dissipation,
    validate_closure,
    add_closure_specific_boundary_conditions,
    calculate_diffusivities!,
    DiffusivityFields,
    implicit_linear_coefficient,
    viscosity,
    diffusivity,
    diffusive_flux_x,
    diffusive_flux_y

struct MEWSVerticalDiffusivity{TD, FT} <: AbstractScalarDiffusivity{TD, VerticalFormulation}
    Cʰ  :: FT
    Cᴷʰ :: FT
    Cᴷᶻ :: FT
    Cⁿ  :: FT
    Cᴰ  :: FT

    function MEWSVerticalDiffusivity{TD}(Cʰ::FT, Cᴷʰ::FT, Cᴷᶻ::FT, Cⁿ::FT, Cᴰ::FT) where {TD, FT}
        return new{TD, FT}(Cʰ, Cᴷʰ, Cᴷᶻ, Cⁿ, Cᴰ)
    end
end

function MEWSVerticalDiffusivity(time_discretization=VerticallyImplicitTimeDiscretization(), FT=Float64;
                                 Cʰ=1, Cᴷʰ=1, Cᴷᶻ=1, Cⁿ=1, Cᴰ=2e-3)
                           
    TD = typeof(time_discretization)

    return MEWSVerticalDiffusivity{TD}(FT(Cʰ), FT(Cᴷʰ), FT(Cᴷᶻ), FT(Cⁿ), FT(Cᴰ))
end

const MEWSVD{TD} = MEWSVerticalDiffusivity{TD} where TD
const MEWSVDArray{TD} = AbstractArray{<:MEWSVD{TD}} where TD
const MEWS{TD} = Union{MEWSVD{TD}, MEWSVDArray{TD}} where TD

required_halo_size(closure::MEWS) = 1 

function with_tracers(tracer_names, closure::MEWS)
    :K ∈ tracer_names ||
        throw(ArgumentError("Tracers must contain :K to represent mesoscale kinetic energy " *
                            "for `MEWSVerticalDiffusivity`."))

    return closure
end

@inline viscosity_location(::MEWS) = (Center(), Center(), Face())
@inline diffusivity_location(::MEWS) = (Center(), Center(), Face())

function DiffusivityFields(grid, tracer_names, bcs, closure::MEWS)
    νₑ = Field{Center, Center, Face}(grid)
    κₖz = Field{Center, Center, Face}(grid)
    κₖh = Field{Center, Center, Center}(grid)
    Lₖ = CenterField(grid)

    # Secret tuple for getting tracer diffusivities with tuple[tracer_index]
    _tupled_tracer_vertical_diffusivities   = NamedTuple(name => name === :K ? κₖz  : ZeroField() for name in tracer_names)
    _tupled_tracer_horizontal_diffusivities = NamedTuple(name => name === :K ? κₖh : ZeroField() for name in tracer_names)
    _tupled_implicit_linear_coefficients    = NamedTuple(name => name === :K ? Lₖ  : ZeroField() for name in tracer_names)

    return (; νₑ, κₖz, κₖh, Lₖ,
            _tupled_tracer_vertical_diffusivities,
            _tupled_tracer_horizontal_diffusivities,
            _tupled_implicit_linear_coefficients)
end

@inline function implicit_linear_coefficient(i, j, k, grid, closure::MEWS, K, ::Val{id}, args...) where id
    L = K._tupled_implicit_linear_coefficients[id]
    return @inbounds L[i, j, k]
end

@inline viscosity(::MEWS, diffusivities) = diffusivities.νₑ
@inline diffusivity(::MEWS, diffusivities, ::Val{id}) where id = diffusivities._tupled_tracer_vertical_diffusivities[id]

@inline bottom_drag_coefficient(closure::MEWSVerticalDiffusivity) = closure.Cᴰ

Base.summary(closure::MEWS) = "MEWSVerticalDiffusivity"

function Base.show(io::IO, closure::MEWS)
    print(io, summary(closure), " with parameters:", '\n',
          string("    Cʰ  : ", closure.Cʰ, '\n',
                 "    Cᴷʰ : ", closure.Cᴷʰ, '\n',
                 "    Cᴷᶻ : ", closure.Cᴷᶻ, '\n',
                 "    Cⁿ  : ", closure.Cⁿ, '\n',
                 "    Cᴰ  : ", closure.Cᴰ))
end

function calculate_diffusivities!(diffusivities, closure::MEWS, model)
    arch = model.architecture
    grid = model.grid
    clock = model.clock
    coriolis = model.coriolis
    tracers = model.tracers
    buoyancy = model.buoyancy
    velocities = model.velocities

    event = launch!(arch, grid, :xyz,
                    compute_mews_diffusivities!,
                    diffusivities,
                    grid,
                    closure,
                    velocities,
                    tracers,
                    buoyancy,
                    coriolis,
                    dependencies = device_event(arch))

    wait(device(arch), event)

    return nothing
end

@inline ϕ²(i, j, k, grid, ϕ, args...) = ϕ(i, j, k, grid, args...)^2

@inline function mews_vertical_displacement(i, j, k, grid, closure, buoyancy, tracers)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    N²⁺ = max(N², zero(grid))
    u★ = ℑzᵃᵃᶠ(i, j, k, grid, mesoscale_turbulent_velocity, tracers.K)
    h★ = ifelse(u★ == 0, zero(grid), u★ / sqrt(N²⁺))

    d = wall_vertical_distanceᶜᶜᶠ(i, j, k, grid)

    Cʰ = closure.Cʰ
    h = min(d, Cʰ * h★)

    # z = znode(Center(), Center(), Face(), i, j, k, grid)
    # h *= (1.0 + abs(cos(π * z / grid.Lz)))

    return h
end

@kernel function compute_mews_diffusivities!(diffusivities, grid, maybe_closure_ensemble,
                                             velocities, tracers, buoyancy, coriolis)

    i, j, k, = @index(Global, NTuple)

    closure = getclosure(i, j, maybe_closure_ensemble)

    # Vertical flux of mesoscale energy
    f = abs(ℑxyᶜᶜᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis))
    f² = ℑxyᶜᶜᵃ(i, j, k, grid, ϕ², fᶠᶠᵃ, coriolis)

    ∂z_u² = ℑxᶜᵃᵃ(i, j, k, grid, ϕ², ∂zᶠᶜᶠ, velocities.u) + ℑyᵃᶜᵃ(i, j, k, grid, ϕ², ∂zᶜᶠᶠ, velocities.v)
    ∂z_u⁻¹ = ifelse(∂z_u² == 0, zero(grid), 1 / sqrt(∂z_u²))

    h = mews_vertical_displacement(i, j, k, grid, closure, buoyancy, tracers)

    u★ = ℑzᵃᵃᶠ(i, j, k, grid, mesoscale_turbulent_velocity, tracers.K)
    ℵ = ∂z_u⁻¹ * u★
    ℵ = min(grid.Lz, ℵ)

    Cⁿ = closure.Cⁿ
    @inbounds diffusivities.νₑ[i, j, k] = Cⁿ * f * h * ℵ

    Cᴷʰ = closure.Cᴷʰ
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    N²⁺ = max(N², zero(grid))

    ℓ = sqrt(N²⁺) * h / f # "eddy length scale"
    @inbounds diffusivities.κₖh[i, j, k] = Cᴷʰ * ℓ * u★

    # Vertical flux of mesoscale energy
    Cᴷᶻ = closure.Cᴷᶻ

    Kᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, clip, tracers.K)
    ∂z_K = abs(∂zᶜᶜᶠ(i, j, k, grid, clip, tracers.K))
    K_∂z_K = ifelse(∂z_K == 0, zero(grid), Kᶜᶜᶠ / ∂z_K)
    K_∂z_K = min(grid.Lz, K_∂z_K)

    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    N²⁺ = max(N², zero(grid))

    # Horizontal flux of mesoscale energy
    Cᴷᶻ = closure.Cᴷᶻ
    @inbounds diffusivities.κₖz[i, j, k] = Cᴷᶻ * f * h * K_∂z_K

    # Bottom drag
    Cᴰ = bottom_drag_coefficient(closure)
    K = @inbounds tracers.K[i, j, k]

    is_bottom = k == 1 # do better.
    K_is_negative = K < 0

    Lₖ = ifelse(is_bottom,     - Cᴰ * sqrt(abs(K)), # time-implicit bottom drag
         ifelse(K_is_negative, - sqrt(abs(K)) / grid.Lz, zero(grid))) # damp spurious K

    # At ccc
    @inbounds diffusivities.Lₖ[i, j, k] = Lₖ
end

@inline function diffusive_flux_x(i, j, k, grid, cl::MEWS, K, ::Val{id}, c, clk, fields, b) where id
    κh = ℑxᶠᵃᵃ(i, j, k, grid, K._tupled_tracer_horizontal_diffusivities[id])
    return - κh * ∂xᶠᶜᶜ(i, j, k, grid, c)
end

@inline function diffusive_flux_y(i, j, k, grid, cl::MEWS, K, ::Val{id}, c, clk, fields, b) where id
    κh = ℑyᵃᶠᵃ(i, j, k, grid, K._tupled_tracer_horizontal_diffusivities[id])
    return - κh * ∂yᶜᶠᶜ(i, j, k, grid, c)
end

#=
@inline νhᶜᶜᶜ(i, j, k, grid, clo::MEWS, K, args...) = @inbounds K.νh[i, j, k]
@inline νhᶠᶠᶜ(i, j, k, grid, clo::MEWS, K, args...) = ℑxyᶠᶠᵃ(i, j, k, grid, K.νh)
@inline νhᶠᶜᶠ(i, j, k, grid, clo::MEWS, K, args...) = ℑxzᶠᵃᶠ(i, j, k, grid, K.νh)
@inline νhᶜᶠᶠ(i, j, k, grid, clo::MEWS, K, args...) = ℑyzᵃᶠᶠ(i, j, k, grid, K.νh)
=#

#=
# Anisotropic viscosity...
# u
@inline νzᶠᶜᶠ(i, j, k, grid, closure::MEWS, K, args...) = @inbounds K.νₑ[i, j, k]
@inline  νᶠᶜᶠ(i, j, k, grid, closure::MEWS, K, args...) = @inbounds K.νₑ[i, j, k]

# v
@inline νzᶜᶠᶠ(i, j, k, grid, closure::MEWS, K, args...) = @inbounds K.νₑ[i, j, k]
@inline  νᶜᶠᶠ(i, j, k, grid, closure::MEWS, K, args...) = @inbounds K.νₑ[i, j, k]
=#

# Mesoscale kinetic energy equation
@inline buoyancy_flux(i, j, k, grid, closure::MEWS, args...) = zero(grid)

@inline ν_∂z_u²(i, j, k, grid, ν, u) = @inbounds ℑxᶠᵃᵃ(i, j, k, grid, ν) * ∂zᶠᶜᶠ(i, j, k, grid, u)^2
@inline ν_∂z_v²(i, j, k, grid, ν, v) = @inbounds ℑyᵃᶠᵃ(i, j, k, grid, ν) * ∂zᶜᶠᶠ(i, j, k, grid, v)^2

@inline function shear_production(i, j, k, grid, closure::MEWS, velocities, diffusivities)
    Pᵤ = ℑxzᶜᵃᶜ(i, j, k, grid, ν_∂z_u², diffusivities.νₑ, velocities.u)
    Pᵥ = ℑyzᵃᶜᶜ(i, j, k, grid, ν_∂z_v², diffusivities.νₑ, velocities.v)

    return Pᵤ + Pᵥ
end

# Treated implicitly
@inline dissipation(i, j, k, grid, closure::MEWS, velocities, tracers, args...) = zero(grid)

@inline clip(x) = max(x, zero(x))
@inline clip(i, j, k, grid, a) = @inbounds clip(a[i, j, k])
@inline mesoscale_turbulent_velocity(i, j, k, grid, K) = @inbounds sqrt(clip(K[i, j, k]))
@inline mke_bottom_dissipation(i, j, grid, clock, fields, Cᴰ) = - Cᴰ * mesoscale_turbulent_velocity(i, j, 1, grid, fields.K)^3

end # module
