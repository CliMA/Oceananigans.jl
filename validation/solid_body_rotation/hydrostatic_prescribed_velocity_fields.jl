#####
##### PrescribedVelocityFields
#####

using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: FunctionField

import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.TimeSteppers: ab2_step_field! 
import Oceananigans.Models.IncompressibleModels: extract_boundary_conditions

import Oceananigans.Models.HydrostaticFreeSurfaceModels:
    HorizontalVelocityFields,
    HydrostaticFreeSurfaceVelocityFields,
    validate_velocity_boundary_conditions,
    compute_w_from_continuity!,
    hydrostatic_prognostic_fields,
    calculate_hydrostatic_momentum_tendencies!

struct PrescribedVelocityFields{U, V, W}
    u :: U
    v :: V
    w :: W
end

@inline Base.getindex(U::PrescribedVelocityFields, i) = getindex((u=U.u, v=U.v, w=U.w), i)

zerofunc(x, y, z) = 0

function PrescribedVelocityFields(grid; u=zerofunc, v=zerofunc, w=zerofunc, parameters=nothing)
    u = FunctionField{Face, Center, Center}(u, grid; parameters=parameters)
    v = FunctionField{Center, Face, Center}(v, grid; parameters=parameters)
    w = FunctionField{Center, Center, Face}(w, grid; parameters=parameters)

    return PrescribedVelocityFields(u, v, w)
end

@inline ab2_step_field!(ϕ::FunctionField, args...) = nothing 
@inline fill_halo_regions!(::PrescribedVelocityFields, args...) = nothing
@inline fill_halo_regions!(::FunctionField, args...) = nothing

extract_boundary_conditions(::PrescribedVelocityFields) = NamedTuple()
HydrostaticFreeSurfaceVelocityFields(velocities::PrescribedVelocityFields, args...) = velocities
FreeSurface(free_surface, ::PrescribedVelocityFields, arch, grid) = nothing
validate_velocity_boundary_conditions(::PrescribedVelocityFields) = nothing
compute_w_from_continuity!(::PrescribedVelocityFields, args...) = nothing
HorizontalVelocityFields(::PrescribedVelocityFields, arch, grid) = nothing, nothing
hydrostatic_prognostic_fields(::PrescribedVelocityFields, free_surface, tracers) = tracers
calculate_hydrostatic_momentum_tendencies!(tendencies, ::PrescribedVelocityFields, args...) = []
