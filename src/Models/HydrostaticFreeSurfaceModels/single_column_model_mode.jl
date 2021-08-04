using KernelAbstractions: NoneEvent
using OffsetArrays: OffsetArray

using Oceananigans.Grids: Flat, Bounded
using Oceananigans.Architectures: device_event

using Oceananigans.BoundaryConditions: fill_bottom_halo!, fill_top_halo!
import Oceananigans.BoundaryConditions: fill_halo_regions!

#####
##### Implements a "single column model mode" for HydrostaticFreeSurfaceModel
#####

const SingleColumnGrid = AbstractGrid{<:AbstractFloat, <:Flat, <:Flat, <:Bounded}

#####
##### Model constructor utils
#####

PressureField(arch, ::SingleColumnGrid) = (pHY′ = nothing,)
FreeSurface(free_surface::ExplicitFreeSurface{Nothing}, velocities, arch, ::SingleColumnGrid) = nothing
FreeSurface(free_surface::ImplicitFreeSurface{Nothing}, velocities, arch, ::SingleColumnGrid) = nothing

validate_momentum_advection(momentum_advection, ::SingleColumnGrid) = nothing
validate_tracer_advection(tracer_advection::AbstractAdvectionScheme, ::SingleColumnGrid) = nothing, NamedTuple()
validate_tracer_advection(tracer_advection::Nothing, ::SingleColumnGrid) = nothing, NamedTuple()

#####
##### Time-step optimizations
#####

calculate_free_surface_tendency!(arch, ::SingleColumnGrid, args...) = NoneEvent()

function calculate_hydrostatic_boundary_tendency_contributions!(Gⁿ, grid::SingleColumnGrid, arch, velocities, free_surface, tracers, args...)

    prognostic_field_names = tuple(:u, :v, propertynames(tracers)...)
    prognostic_fields = merge(velocities, tracers)
    barrier = device_event(arch)

    # Only apply z bcs
    events = Tuple(apply_z_bcs!(Gⁿ[i], prognostic_fields[i], arch, barrier, args...) for i in prognostic_field_names)

    wait(device(arch), MultiEvent(events))

    return nothing
end

function update_state!(model::HydrostaticFreeSurfaceModel, grid::SingleColumnGrid)

    # Fill halos for velocities and tracers. On the CubedSphere, the halo filling for velocity fields is wrong.
    fill_halo_regions!(prognostic_fields(model), model.architecture, model.clock, fields(model))

    compute_auxiliary_fields!(model.auxiliary_fields)

    # Calculate diffusivities
    calculate_diffusivities!(model.diffusivity_fields,
                             model.architecture,
                             model.grid,
                             model.closure,
                             model.buoyancy,
                             model.velocities,
                             model.tracers)

    fill_halo_regions!(model.diffusivity_fields,
                       model.architecture,
                       model.clock,
                       fields(model))

    return nothing
end

function fill_halo_regions!(c::OffsetArray, bcs, arch::CPU, grid::SingleColumnGrid, args...; kwargs...)

    wait(device(arch), device_event(arch))

    fill_scm_bottom_halo!(c, grid, bcs.bottom, args...)
    fill_scm_top_halo!(c, grid, bcs.top, args...)

    return nothing
end

using Oceananigans.Operators: Δzᵃᵃᶜ
using Oceananigans.BoundaryConditions: left_gradient, right_gradient, linearly_extrapolate, FBC, VBC, GBC
    
@inline fill_scm_bottom_halo!(c, grid, ::FBC, args...) = @inbounds c[1, 1, 0] = c[1, 1, 1]
@inline fill_scm_top_halo!(c, grid, ::FBC, args...) = @inbounds c[1, 1, grid.Nz+1] = c[1, 1, grid.Nz]
        
@inline function fill_scm_top_halo!(c, grid, bc::Union{VBC, GBC}, args...)
    i = j = 1

    kᴴ = grid.Nz + 1 #    *    halo cell
    kᴮ = grid.Nz + 1 #  =====  top boundary 
    kᴵ = grid.Nz     #    *    interior cell

    Δ = Δzᵃᵃᶜ(i, j, kᴮ, grid) # Δ between first interior and first top halo point, defined at cell face.
    @inbounds ∇c = right_gradient(bc, c[i, j, kᴵ], Δ, i, j, grid, args...)
    @inbounds c[i, j, kᴴ] = linearly_extrapolate(c[i, j, kᴵ], ∇c, Δ) # extrapolate upward in +z direction.
end

@inline function fill_scm_bottom_halo!(c, grid, bc::Union{VBC, GBC}, args...)

    i = j = 1

        #  ↑ z ↑  interior
           #  -----  interior face
    kᴵ = 1 #    *    interior cell
    kᴮ = 1 #  =====  bottom boundary
    kᴴ = 0 #    *    halo cell

    Δ = Δzᵃᵃᶜ(i, j, kᴮ, grid) # Δ between first interior and first bottom halo point, defined at cell face.
    @inbounds ∇c = left_gradient(bc, c[i, j, kᴵ], Δ, i, j, grid, args...)
    @inbounds c[i, j, kᴴ] = linearly_extrapolate(c[i, j, kᴵ], ∇c, -Δ) # extrapolate downward in -z direction.

    return nothing
end
