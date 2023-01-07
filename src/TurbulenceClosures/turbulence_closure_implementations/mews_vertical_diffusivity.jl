module MEWSVerticalDiffusivities

import Oceananigans.Grids: required_halo_size
using Oceananigans.Utils: prettysummary

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
with_tracers(tracers, closure::MEWS) = closure

@inline viscosity_location(::MEWS) = (Center(), Center(), Face())
@inline diffusivity_location(::MEWS) = (Center(), Center(), Face())

function DiffusivityFields(grid, tracer_names, bcs, closure::MEWS)
    νᵤ = Field{Face, Center, Face}(grid)
    νᵥ = Field{Center, Face, Face}(grid)
    νₖ = Field{Center, Center, Face}(grid)

    # Secret tuple for getting tracer diffusivities with tuple[tracer_index]
    _tupled_tracer_diffusivities = NamedTuple(name => name === :K ? νₖ : ZeroField() for name in tracer_names)

    return (; νᵤ, νₖ, _tupled_tracer_diffusivities)
end        

# @inline viscosity(closure::MEWS, K) = K.νᵤ # do we need this?
@inline diffusivity(::MEWS, diffusivities, ::Val{id}) where id = diffusivities._tupled_tracer_diffusivities[id]

calculate_diffusivities!(diffusivities, ::MEWS, args...) = nothing

Base.summary(closure::MEWS) = "MEWSVerticalDiffusivity"
Base.show(io::IO, closure::MEWS) = print(io, summary(closure))

# Anisotropic viscosity...
# u
@inline νzᶠᶜᶠ(i, j, k, grid, closure::MEWS, K, args...) = @inbounds K.νᵤ[i, j, k]
@inline  νᶠᶜᶠ(i, j, k, grid, closure::MEWS, K, args...) = @inbounds K.νᵤ[i, j, k]

# v
@inline νzᶜᶠᶠ(i, j, k, grid, closure::MEWS, K, args...) = @inbounds K.νᵥ[i, j, k]
@inline  νᶜᶠᶠ(i, j, k, grid, closure::MEWS, K, args...) = @inbounds K.νᵥ[i, j, k]

# Mesoscale kinetic energy equation
@inline buoyancy_flux(i, j, k, grid, closure::MEWS, args...) = zero(grid)

@inline ν_uz²(i, j, k, grid, νu, u) = @inbounds νu[i, j, k] * ∂zᶠᶜᶠ(i, j, k, grid, u)^2
@inline ν_vz²(i, j, k, grid, νv, v) = @inbounds νv[i, j, k] * ∂zᶜᶠᶠ(i, j, k, grid, v)^2

@inline function mews_shear_productionᶜᶜᶠ(i, j, k, grid, velocities, diffusivities)
    ∂z_u² = ∂zᶠᶜᶠ(i, j, k, grid, velocities.u)^2
    ∂z_v² = ∂zᶜᶠᶠ(i, j, k, grid, velocities.v)^2

    Pᵤ = ℑxᶜᵃᵃ(i, j, k, grid, ν_uz², diffusivities.νᵤ, velocities.u)
    Pᵥ = ℑyᵃᶜᵃ(i, j, k, grid, ν_vz², diffusivities.νᵥ, velocities.v)

    return Pᵤ + Pᵥ
end

@inline function shear_production(i, j, k, grid, closure::MEWS, velocities, diffusivities) =
    ℑzᵃᵃᶜ(i, j, k, grid, mews_shear_productionᶜᶜᶠ, velocities, diffusivities)

@inline dissipation(i, j, k, grid, closure::MEWS, args...) = zero(grid)

@inline clip(x) = max(x, zero(x))
@inline mke_bottom_dissipation(i, j, grid, clock, fields, Cᴰ) = - Cᴰ * sqrt(clip(fields.K[i, j, 1]))^3

bottom_drag_coefficient(closure::MEWSVerticalDiffusivity) = closure.Cᴰ

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
                                          bottom = bottom_mke_bc
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

end # module
