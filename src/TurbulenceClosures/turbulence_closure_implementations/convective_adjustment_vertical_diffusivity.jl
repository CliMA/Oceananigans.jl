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
    ConvectiveAdjustmentVerticalDiffusivity([FT=Float64;]
                                            convective_κz = 0,
                                            convective_νz = 0,
                                            background_κz = 0,
                                            background_νz = 0,
                                            time_discretization = VerticallyImplicitTimeDiscretization())

The one positional argument determines the floating point type of the free parameters
of `ConvectiveAdjustmentVerticalDiffusivity`. The default is `Float64`.

Keyword arguments
=================

* `convective_κz`: Vertical tracer diffusivity in regions with negative (unstable) buoyancy gradients. Either
                   a single number, function, array, field, or tuple of diffusivities for each tracer.

* `background_κz`: Vertical tracer diffusivity in regions with zero or positive (stable) buoyancy gradients.

* `convective_νz`: Vertical viscosity in regions with negative (unstable) buoyancy gradients. Either
                  a number, function, array, or field.

* `background_κz`: Vertical viscosity in regions with zero or positive (stable) buoyancy gradients.

* `time_discretization`: Either `ExplicitTimeDiscretization` or `VerticallyImplicitTimeDiscretization`.
"""
function ConvectiveAdjustmentVerticalDiffusivity(FT = Float64;
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

with_tracers(tracers, closure::ConvectiveAdjustmentVerticalDiffusivity{TD}) where TD =
    ConvectiveAdjustmentVerticalDiffusivity{TD}(closure.convective_κz,
                                                closure.convective_νz,
                                                closure.background_κz,
                                                closure.background_νz)

# Note: computing diffusivities at cell centers for now.
function DiffusivityFields(arch, grid, tracer_names, bcs, closure::CAVD)
    #data = new_data(Bool, arch, grid, (Center, Center, Center))
    κ = Field(Center, Center, Center, arch, grid)
    ν = Field(Center, Center, Center, arch, grid)
    return (; κ, ν)
end       

function calculate_diffusivities!(diffusivities, closure::CAVD, model)

    arch = model.architecture
    grid = model.grid
    tracers = model.tracers
    buoyancy = model.buoyancy

    event = launch!(arch, grid, :xyz,
                    #compute_stability!, diffusivities, grid, closure, tracers, buoyancy,
                    compute_convective_adjustment_diffusivities!, diffusivities, grid, closure, tracers, buoyancy,
                    dependencies = device_event(arch))

    wait(device(arch), event)

    return nothing
end

@inline is_unstableᶜᶜᶠ(i, j, k, grid, tracers, buoyancy) = ∂z_b(i, j, k, grid, buoyancy, tracers) < 0

@kernel function compute_convective_adjustment_diffusivities!(diffusivities, grid, closure, tracers, buoyancy)
    i, j, k, = @index(Global, NTuple)

    unstable_cell = is_unstableᶜᶜᶠ(i, j, k+1, grid, tracers, buoyancy) | 
                    is_unstableᶜᶜᶠ(i, j, k,   grid, tracers, buoyancy)

    @inbounds diffusivities.κ[i, j, k] = ifelse(unstable_cell,
                                                closure.convective_κz,
                                                closure.background_κz)

    @inbounds diffusivities.ν[i, j, k] = ifelse(unstable_cell,
                                                closure.convective_νz,
                                                closure.background_νz)
end

#=
@kernel function compute_stability!(diffusivities, grid, closure, tracers, buoyancy)
    i, j, k, = @index(Global, NTuple)
    @inbounds diffusivities.unstable_buoyancy_gradient[i, j, k] = is_unstableᶜᶜᶠ(i, j, k, grid, tracers, buoyancy)
end
=#

#####
##### Fluxes
#####

const VITD = VerticallyImplicitTimeDiscretization
const ATD = AbstractTimeDiscretization

@inline viscous_flux_ux(i, j, k, grid, ::ATD, closure::CAVD, args...) = zero(eltype(grid))
@inline viscous_flux_uy(i, j, k, grid, ::ATD, closure::CAVD, args...) = zero(eltype(grid))
@inline viscous_flux_vx(i, j, k, grid, ::ATD, closure::CAVD, args...) = zero(eltype(grid))
@inline viscous_flux_vy(i, j, k, grid, ::ATD, closure::CAVD, args...) = zero(eltype(grid))
@inline viscous_flux_wx(i, j, k, grid, ::ATD, closure::CAVD, args...) = zero(eltype(grid))
@inline viscous_flux_wy(i, j, k, grid, ::ATD, closure::CAVD, args...) = zero(eltype(grid))

@inline diffusive_flux_x(i, j, k, grid, ::ATD, closure::CAVD, args...) = zero(eltype(grid))
@inline diffusive_flux_y(i, j, k, grid, ::ATD, closure::CAVD, args...) = zero(eltype(grid))

@inline viscous_flux_ux(i, j, k, grid, closure::CAVD, args...) = zero(eltype(grid))
@inline viscous_flux_uy(i, j, k, grid, closure::CAVD, args...) = zero(eltype(grid))
@inline viscous_flux_vx(i, j, k, grid, closure::CAVD, args...) = zero(eltype(grid))
@inline viscous_flux_vy(i, j, k, grid, closure::CAVD, args...) = zero(eltype(grid))
@inline viscous_flux_wx(i, j, k, grid, closure::CAVD, args...) = zero(eltype(grid))
@inline viscous_flux_wy(i, j, k, grid, closure::CAVD, args...) = zero(eltype(grid))

@inline diffusive_flux_x(i, j, k, grid, closure::CAVD, args...) = zero(eltype(grid))
@inline diffusive_flux_y(i, j, k, grid, closure::CAVD, args...) = zero(eltype(grid))

#####
##### Diffusivity
#####

const etd = ExplicitTimeDiscretization()

@inline z_boundary_adj(k, grid) = k == 1 | k == grid.Nz+1

@inline z_diffusivity(closure::CAVD, c_idx, diffusivities, args...) = diffusivities.κ

@inline function diffusive_flux_z(i, j, k, grid, closure::CAVD, c, tracer_index, clock, diffusivities, args...)
    κ = κᶜᶜᶠ(i, j, k, grid, clock, diffusivities.κ)
    return - κ * ∂zᵃᵃᶠ(i, j, k, grid, c)
end

@inline function diffusive_flux_z(i, j, k, grid::VerticallyBoundedGrid, ::VITD, closure::CAVD, args...)
    explicit_flux_z = diffusive_flux_z(i, j, k, grid, etd, closure, args...)
    return ifelse(z_boundary_adj(k, grid), explicit_flux_z, zero(eltype(grid)))
end
 
#####
##### Viscosity
#####

@inline z_viscosity(closure::CAVD, diffusivities, args...) = diffusivities.ν

@inline function viscous_flux_uz(i, j, k, grid::VerticallyBoundedGrid, ::VITD, closure::CAVD, args...)
    explicit_flux_z = viscous_flux_uz(i, j, k, grid, etd, closure, args...)
    return ifelse(z_boundary_adj(k, grid), explicit_flux_z, zero(eltype(grid)))
end

@inline function viscous_flux_vz(i, j, k, grid::VerticallyBoundedGrid, ::VITD, closure::CAVD, args...)
    explicit_flux_z = viscous_flux_vz(i, j, k, grid, etd, closure, args...)
    return ifelse(z_boundary_adj(k, grid), explicit_flux_z, zero(eltype(grid)))
end

@inline function viscous_flux_wz(i, j, k, grid::VerticallyBoundedGrid, ::VITD, closure::CAVD, args...)
    explicit_flux_z = viscous_flux_wz(i, j, k, grid, etd, closure, args...)
    return ifelse(z_boundary_adj(k, grid), explicit_flux_z, zero(eltype(grid)))
end

@inline function viscous_flux_uz(i, j, k, grid, closure::CAVD, clock, velocities, diffusivities, args...)
    ν = νᶠᶜᶠ(i, j, k, grid, clock, diffusivities.ν)
    return - ν * ∂zᵃᵃᶠ(i, j, k, grid, velocities.u)
end

@inline function viscous_flux_vz(i, j, k, grid, closure::CAVD, clock, velocities, diffusivities, args...)
    ν = νᶜᶠᶠ(i, j, k, grid, clock, diffusivities.ν)
    return - ν * ∂zᵃᵃᶠ(i, j, k, grid, velocities.v)
end

@inline function viscous_flux_wz(i, j, k, grid, closure::CAVD, clock, velocities, diffusivities, args...)
    ν = νᶜᶜᶜ(i, j, k, grid, clock, diffusivities.ν)
    return - ν * ∂zᵃᵃᶜ(i, j, k, grid, velocities.w)
end

#####
##### Show
#####
Base.show(io::IO, closure::ConvectiveAdjustmentVerticalDiffusivity) =
    print(io, "ConvectiveAdjustmentVerticalDiffusivity: " *
              "(background_κz=$(closure.background_κz), convective_κz=$(closure.convective_κz), " *
              "background_νz=$(closure.background_νz), convective_νz=$(closure.convective_νz)" * ")")
