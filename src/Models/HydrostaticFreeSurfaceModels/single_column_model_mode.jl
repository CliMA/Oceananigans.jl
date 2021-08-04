using KernelAbstractions: NoneEvent
using OffsetArrays: OffsetArray

using Oceananigans.Operators: Δzᵃᵃᶜ
using Oceananigans.BoundaryConditions: left_gradient, right_gradient, linearly_extrapolate, FBC, VBC, GBC
using Oceananigans.BoundaryConditions: fill_bottom_halo!, fill_top_halo!, apply_z_bottom_bc!, apply_z_top_bc!
using Oceananigans.Grids: Flat, Bounded
using Oceananigans.Architectures: device_event

import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.Utils: launch!

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

# Fast flux calculation

@inline function calculate_hydrostatic_boundary_tendency_contributions!(Gⁿ,
                                                                        grid::SingleColumnGrid,
                                                                        arch::CPU,
                                                                        velocities,
                                                                        free_surface,
                                                                        tracers,
                                                                        args...)

    prognostic_field_names = tuple(:u, :v, propertynames(tracers)...)
    prognostic_fields = merge(velocities, tracers)

    for name in prognostic_field_names
        @inbounds begin
            Gcⁿ = Gⁿ[name]
            c = prognostic_fields[name]
        end
        loc = (L() for L in location(c))
        apply_scm_z_bcs!(Gcⁿ, grid, c.boundary_conditions.bottom, c.boundary_conditions.top, loc, args...)
    end

    return nothing
end

@inline function apply_scm_z_bcs!(Gc, grid, bottom_bc, top_bc, loc, args...)
    i = j = 1
    apply_z_bottom_bc!(Gc, loc, bottom_bc, i, j, grid, args...)
       apply_z_top_bc!(Gc, loc, top_bc,    i, j, grid, args...)
    return nothing
end

# Fast state update and halo filling

function update_state!(model::HydrostaticFreeSurfaceModel, grid::SingleColumnGrid)

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

@inline function fill_halo_regions!(c::OffsetArray, bcs, arch::CPU, grid::SingleColumnGrid, args...; kwargs...)
    fill_scm_bottom_halo!(c, grid, bcs.bottom, args...)
    fill_scm_top_halo!(c, grid, bcs.top, args...)
    return nothing
end
    
@inline fill_scm_bottom_halo!(c, grid, ::FBC, args...) = @inbounds c[1, 1, 0] = c[1, 1, 1]
@inline fill_scm_top_halo!(c, grid, ::FBC, args...) = @inbounds c[1, 1, grid.Nz+1] = c[1, 1, grid.Nz]
        
@inline function fill_scm_top_halo!(c, grid, bc::Union{VBC, GBC}, args...)
    i = j = 1

                     #  ↑ z ↑
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

# Fast kernel launching... ?

@inline function launch!(arch::CPU, grid::SingleColumnGrid, dims, kernel!, args...;
                         dependencies = nothing,
                         include_right_boundaries = false,
                         location = nothing)

    workgroup = (1, 1, grid.Nz)
    worksize = (1, 1, grid.Nz)
    loop! = kernel!(Architectures.device(arch), workgroup, worksize)
    event = loop!(args...; dependencies)
    return event
end
