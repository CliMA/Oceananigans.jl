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
    diffusivity

struct MEWSVerticalDiffusivity{TD, FT} <: AbstractScalarDiffusivity{TD, VerticalFormulation}
    Cʰ :: FT
    Cᴷ :: FT
    Cⁿ :: FT
    Cᴰ :: FT

    function MEWSVerticalDiffusivity{TD}(Cʰ::FT, Cᴷ::FT, Cⁿ::FT, Cᴰ::FT) where {TD, FT}
        return new{TD, FT}(Cʰ, Cᴷ, Cⁿ, Cᴰ)
    end
end

function MEWSVerticalDiffusivity(time_discretization=VerticallyImplicitTimeDiscretization(), FT=Float64;
                                 Cʰ=1, Cᴷ=1, Cⁿ=1, Cᴰ=2e-3)
                           
    TD = typeof(time_discretization)

    return MEWSVerticalDiffusivity{TD}(FT(Cʰ), FT(Cᴷ), FT(Cⁿ), FT(Cᴰ))
end

const MEWS = MEWSVerticalDiffusivity

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
    νₖ = Field{Center, Center, Face}(grid)
    Lₖ = CenterField(grid)

    # Secret tuple for getting tracer diffusivities with tuple[tracer_index]
    _tupled_tracer_diffusivities         = NamedTuple(name => name === :K ? νₖ : ZeroField() for name in tracer_names)
    _tupled_implicit_linear_coefficients = NamedTuple(name => name === :K ? Lₖ : ZeroField() for name in tracer_names)

    return (; νₑ, νₖ, Lₖ, _tupled_tracer_diffusivities, _tupled_implicit_linear_coefficients)
end

@inline function implicit_linear_coefficient(i, j, k, grid, closure::MEWS, K, ::Val{id}, args...) where id
    L = K._tupled_implicit_linear_coefficients[id]
    return @inbounds L[i, j, k]
end

@inline viscosity(::MEWS, diffusivities) = diffusivities.νₑ
@inline diffusivity(::MEWS, diffusivities, ::Val{id}) where id = diffusivities._tupled_tracer_diffusivities[id]

calculate_diffusivities!(diffusivities, ::MEWS, args...) = nothing

@inline bottom_drag_coefficient(closure::MEWSVerticalDiffusivity) = closure.Cᴰ

Base.summary(closure::MEWS) = "MEWSVerticalDiffusivity"
Base.show(io::IO, closure::MEWS) = print(io, summary(closure))

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
    h = min(10 * grid.Lz, Cʰ * h★)
    #h = Cʰ * h★

    return h
end

@kernel function compute_mews_diffusivities!(diffusivities, grid, closure,
                                             velocities, tracers, buoyancy, coriolis)

    i, j, k, = @index(Global, NTuple)

    # Vertical flux of mesoscale energy
    f = abs(ℑxyᶜᶜᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis))
    f² = ℑxyᶜᶜᵃ(i, j, k, grid, ϕ², fᶠᶠᵃ, coriolis)

    bx = ℑxzᶜᵃᶠ(i, j, k, grid, ∂x_b, buoyancy, tracers)
    by = ℑyzᵃᶜᶠ(i, j, k, grid, ∂y_b, buoyancy, tracers)
    ∇h_b = sqrt(bx^2 + by^2)
    ∇h_b⁻¹ = ifelse(∇h_b == 0, zero(grid), 1 / ∇h_b)

    ∂z_u² = ℑxᶜᵃᵃ(i, j, k, grid, ϕ², ∂zᶠᶜᶠ, velocities.u) + ℑyᵃᶜᵃ(i, j, k, grid, ϕ², ∂zᶜᶠᶠ, velocities.v)
    ∂z_u⁻¹ = ifelse(∂z_u² == 0, zero(grid), 1 / sqrt(∂z_u²))

    h = mews_vertical_displacement(i, j, k, grid, closure, buoyancy, tracers)

    u★ = ℑzᵃᵃᶠ(i, j, k, grid, mesoscale_turbulent_velocity, tracers.K)
    ℵ = u★ * ∂z_u⁻¹
    ℵ = min(grid.Lz, ℵ)

    Cⁿ = closure.Cⁿ
    @inbounds diffusivities.νₑ[i, j, k] = Cⁿ * f * h * ℵ

    # Vertical flux of mesoscale energy
    Cᴷ = closure.Cᴷ

    Kᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, clip, tracers.K)
    ∂z_K = abs(∂zᶜᶜᶠ(i, j, k, grid, clip, tracers.K))
    K_∂z_K = ifelse(∂z_K == 0, zero(grid), Kᶜᶜᶠ / ∂z_K)
    K_∂z_K = min(grid.Lz, K_∂z_K)

    @inbounds diffusivities.νₖ[i, j, k] = Cᴷ * f * h * K_∂z_K

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

#=
""" Add MEKE boundary conditions specific to `MEWSVerticalDiffusivity`. """
function add_closure_specific_boundary_conditions(closure::MEWS,
                                                  user_bcs,
                                                  grid,
                                                  tracer_names,
                                                  buoyancy)

    Cᴰ = bottom_drag_coefficient(closure)
    bottom_mke_bc = FluxBoundaryCondition(mke_bottom_dissipation, discrete_form=true, parameters=Cᴰ)

    if :K ∈ keys(user_bcs)
        K_bcs = user_bcs[:K]
        
        mke_bcs = FieldBoundaryConditions(grid, (Center, Center, Center),
                                          top    = K_bcs.top,
                                          bottom = bottom_mke_bc,
                                          north  = K_bcs.north,
                                          south  = K_bcs.south,
                                          east   = K_bcs.east,
                                          west   = K_bcs.west)
    else
        mke_bcs = FieldBoundaryConditions(grid, (Center, Center, Center), bottom=bottom_mke_bc)
    end

    new_boundary_conditions = merge(user_bcs, (; K = mke_bcs))

    return new_boundary_conditions
end
=#

end # module
