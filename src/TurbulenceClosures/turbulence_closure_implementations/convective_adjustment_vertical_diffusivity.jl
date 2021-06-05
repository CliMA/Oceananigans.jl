using Oceananigans.Architectures: architecture, device_event
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.BuoyancyModels: ∂z_b
using Oceananigans.Operators: ℑzᵃᵃᶜ

struct ConvectiveAdjustmentVerticalDiffusivity{TD, CK, CN, BK, BN} <: AbstractTurbulenceClosure{TD}
    convective_κz :: CK
    convective_νz :: CN
    background_κz :: BK
    background_νz :: BN

    function ConvectiveAdjustmentVerticalDiffusivity{TD}(convective_κz::CK,
                                                         convective_νz::CN,
                                                         background_κz::BK,
                                                         background_νz::BN) where {TD, CK, CN, BK, BN}

        return new{TD, CK, CN, BK, BN}(convective_κz, convective_νz, background_κz, background_νz)
    end
end
                                       
"""
    ConvectiveAdjustmentVerticalDiffusivity(FT=Float64; convective_κz=0, convective_νz=0, background_κz=0, background_νz=0,
                                            time_discretization = VerticallyImplicitTimeDiscretization())

The one positional argument determines the floating point type of the free parameters
of `ConvectiveAdjustmentVerticalDiffusivity`. The default is `Float64`.

Keyword arguments
=================

* `convective_κz` : Vertical tracer diffusivity in regions with unstable buoyancy gradients. Either
                    a single number, function, array, field, or tuple of diffusivities for each tracer.

* `background_κz` : Vertical tracer diffusivity in regions with stable buoyancy gradients.

* `convective_νz` : Vertical viscosity in regions with unstable buoyancy gradients. Either
                    a number, function, array, or field.

* `background_κz` : Vertical viscosity in regions with stable buoyancy gradients.

* `time_discretization` : Either `ExplicitTimeDiscretization` or `VerticallyImplicitTimeDiscretization`.
"""
function ConvectiveAdjustmentVerticalDiffusivity(FT=Float64;
                                                 convective_κz = zero(FT),
                                                 convective_νz = zero(FT),
                                                 background_κz = zero(FT),
                                                 background_νz = zero(FT),
                                                 time_discretization::TD = VerticallyImplicitTimeDiscretization()) where TD

    return ConvectiveAdjustmentVerticalDiffusivity{TD}(convective_κz, convective_νz,
                                                       background_κz, background_νz)
end

const CAVD = ConvectiveAdjustmentVerticalDiffusivity

#####
##### Diffusivity field utilities
#####

function with_tracers(tracers, closure::ConvectiveAdjustmentVerticalDiffusivity{TD}) where TD
    background_κz = tracer_diffusivities(tracers, closure.background_κz)
    convective_κz = tracer_diffusivities(tracers, closure.convective_κz)
    return ConvectiveAdjustmentVerticalDiffusivity{TD}(convective_κz,
                                                       closure.convective_νz,
                                                       background_κz,
                                                       closure.background_νz)
end


function DiffusivityFields(arch, grid, tracer_names, bcs, closure::CAVD)
    data = new_data(Bool, arch, grid, (Center, Center, Face))
    stable_buoyancy_gradient = Field(Center, Center, Face, arch, grid, nothing, data)
    return (; stable_buoyancy_gradient)
end       

function calculate_diffusivities!(diffusivities, arch, grid, closure::CAVD, buoyancy, velocities, tracers)

    event = launch!(arch, grid, :xyz,
                    compute_stability!, diffusivities, grid, tracers, buoyancy,
                    dependencies = device_event(arch))

    wait(device(arch), event)

    return nothing
end

@inline is_stableᶜᶜᶠ(i, j, k, grid, tracers, buoyancy) = ∂z_b(i, j, k, grid, buoyancy, tracers) > 0

@kernel function compute_stability!(diffusivities, grid, tracers, buoyancy)
    i, j, k, = @index(Global, NTuple)
    @inbounds  diffusivities.stable_buoyancy_gradient[i, j, k] = is_stableᶜᶜᶠ(i, j, k, grid, tracers, buoyancy)
end

#####
##### Fluxes
#####

# u
@inline function νᶠᶜᶠ(i, j, k, grid, clock, closure::CAVD, stable_buoyancy_gradient)
    @inbounds stableᶠᶜᶠ = stable_buoyancy_gradient[i, j, k] || stable_buoyancy_gradient[i+1, j, k]

    ν = ifelse(stableᶠᶜᶠ,
               νᶠᶜᶠ(i, j, k, grid, clock, closure.background_νz),
               νᶠᶜᶠ(i, j, k, grid, clock, closure.convective_νz))

    return ν
end

@inline function viscous_flux_uz(i, j, k, grid, closure::CAVD, clock, velocities, diffusivities, tracers, buoyancy)
    ν = νᶠᶜᶠ(i, j, k, grid, clock, closure, diffusivities.stable_buoyancy_gradient)
    return - ν * ∂zᵃᵃᶠ(i, j, k, grid, velocities.u)
end

# v
@inline function νᶜᶠᶠ(i, j, k, grid, clock, closure::CAVD, stable_buoyancy_gradient)
    @inbounds stableᶜᶠᶠ = stable_buoyancy_gradient[i, j, k] || stable_buoyancy_gradient[i, j+1, k]

    ν = ifelse(stableᶜᶠᶠ,
               νᶜᶠᶠ(i, j, k, grid, clock, closure.background_νz),
               νᶜᶠᶠ(i, j, k, grid, clock, closure.convective_νz))

    return ν
end

@inline function viscous_flux_vz(i, j, k, grid, closure::CAVD, clock, velocities, diffusivities, tracers, buoyancy)
    ν = νᶜᶠᶠ(i, j, k, grid, clock, closure, diffusivities.stable_buoyancy_gradient)
    return - ν * ∂zᵃᵃᶠ(i, j, k, grid, velocities.v)
end

# w
@inline function νᶜᶜᶜ(i, j, k, grid, clock, closure::CAVD, stable_buoyancy_gradient)
    @inbounds stableᶜᶜᶜ = stable_buoyancy_gradient[i, j, k] || stable_buoyancy_gradient[i, j, k+1]

    ν = ifelse(stableᶜᶜᶜ,
               νᶜᶜᶜ(i, j, k, grid, clock, closure.background_νz),
               νᶜᶜᶜ(i, j, k, grid, clock, closure.convective_νz))

    return ν
end

@inline function viscous_flux_wz(i, j, k, grid, closure::CAVD, clock, velocities, diffusivities, tracers, buoyancy)
    ν = νᶜᶜᶜ(i, j, k, grid, clock, closure, diffusivities.stable_buoyancy_gradient)
    return - ν * ∂zᵃᵃᶜ(i, j, k, grid, velocities.w)
end

# tracers
@inline function κᶜᶜᶠ(i, j, k, grid, clock, closure::CAVD, stable_buoyancy_gradient, ::Val{tracer_index}) where tracer_index
    @inbounds stableᶜᶜᶠ = stable_buoyancy_gradient[i, j, k]

    background_κz = closure.background_κz[tracer_index]
    convective_κz = closure.convective_κz[tracer_index]

    κ = ifelse(stableᶜᶜᶠ,
               κᶜᶜᶠ(i, j, k, grid, clock, background_κz),
               κᶜᶜᶠ(i, j, k, grid, clock, convective_κz))

    return κ
end

@inline function diffusive_flux_z(i, j, k, grid, closure::CAVD, c, tracer_index, clock, diffusivities, tracers, buoyancy, velocities)
    κ = κᶜᶜᶠ(i, j, k, grid, clock, closure, diffusivities.stable_buoyancy_gradient, tracer_index)
    return - κ * ∂zᵃᵃᶠ(i, j, k, grid, c)
end

#####
##### Support for VerticallyImplicitTimeDiscretization
#####

struct ConvectiveAdjustmentCoeff{I, C, U}
    closure :: C
    stable_buoyancy_gradient :: U
    function ConvectiveAdjustmentCoeff{I}(closure::C, stable_buoyancy_gradient::U) where {I, C, U}
        return new{I, C, U}(closure, stable_buoyancy_gradient)
    end
end

@inline z_viscosity(closure::CAVD, diffusivities, args...) =
    ConvectiveAdjustmentCoeff{Nothing}(closure, diffusivities.stable_buoyancy_gradient)

@inline z_diffusivity(closure::CAVD, ::Val{tracer_index}, diffusivities, args...) where tracer_index =
    ConvectiveAdjustmentCoeff{tracer_index}(closure, diffusivities.stable_buoyancy_gradient)

@inline νᶜᶜᶜ(i, j, k, grid, clock, c::ConvectiveAdjustmentCoeff) = νᶜᶜᶜ(i, j, k, grid, clock, c.closure, c.stable_buoyancy_gradient)
@inline νᶠᶠᶜ(i, j, k, grid, clock, c::ConvectiveAdjustmentCoeff) = νᶠᶠᶜ(i, j, k, grid, clock, c.closure, c.stable_buoyancy_gradient)
@inline νᶠᶜᶠ(i, j, k, grid, clock, c::ConvectiveAdjustmentCoeff) = νᶠᶜᶠ(i, j, k, grid, clock, c.closure, c.stable_buoyancy_gradient)
@inline νᶜᶠᶠ(i, j, k, grid, clock, c::ConvectiveAdjustmentCoeff) = νᶜᶠᶠ(i, j, k, grid, clock, c.closure, c.stable_buoyancy_gradient)
@inline κᶜᶜᶠ(i, j, k, grid, clock, c::ConvectiveAdjustmentCoeff{I}) where I = κᶜᶜᶠ(i, j, k, grid, clock, c.closure, c.stable_buoyancy_gradient, Val(I))

const VITD = VerticallyImplicitTimeDiscretization
const APG = AbstractPrimaryGrid
const VerticallyBoundedGrid{FT} = AbstractPrimaryGrid{FT, <:Any, <:Any, <:Bounded}

@inline diffusive_flux_z(i, j, k, grid::APG{FT}, ::VITD, closure::CAVD, args...) where FT = zero(FT)
@inline viscous_flux_uz(i, j, k, grid::APG{FT}, ::VITD, closure::CAVD, args...) where FT = zero(FT)
@inline viscous_flux_vz(i, j, k, grid::APG{FT}, ::VITD, closure::CAVD, args...) where FT = zero(FT)

@inline function diffusive_flux_z(i, j, k, grid::VerticallyBoundedGrid{FT}, ::VITD, closure::CAVD, args...) where FT
    return ifelse(k == 1 || k == grid.Nz+1, 
                  diffusive_flux_z(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...), # on boundaries, calculate fluxes explicitly
                  zero(FT))
end

@inline function viscous_flux_uz(i, j, k, grid::VerticallyBoundedGrid{FT}, ::VITD, closure::CAVD, args...) where FT
    return ifelse(k == 1 || k == grid.Nz+1, 
                  viscous_flux_uz(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...), # on boundaries, calculate fluxes explicitly
                  zero(FT))
end

@inline function viscous_flux_vz(i, j, k, grid::VerticallyBoundedGrid{FT}, ::VITD, closure::CAVD, args...) where FT
    return ifelse(k == 1 || k == grid.Nz+1, 
                  viscous_flux_vz(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...), # on boundaries, calculate fluxes explicitly
                  zero(FT))
end

@inline function viscous_flux_wz(i, j, k, grid::VerticallyBoundedGrid{FT}, ::VITD, closure::CAVD, args...) where FT
    return ifelse(k == 1 || k == grid.Nz+1, 
                  viscous_flux_wz(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...), # on boundaries, calculate fluxes explicitly
                  zero(FT))
end

