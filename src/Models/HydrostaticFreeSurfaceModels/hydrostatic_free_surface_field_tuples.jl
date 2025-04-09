using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: XFaceField, YFaceField, ZFaceField, TracerFields
using Oceananigans.TimeSteppers: QuasiAdamsBashforth2TimeStepper, SplitRungeKutta3TimeStepper
using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: QuasiAdamsBashforth2TimeStepper, SplitRungeKutta3TimeStepper

function hydrostatic_velocity_fields(::Nothing, grid, clock, bcs)
    u = XFaceField(grid, boundary_conditions=bcs.u)
    v = YFaceField(grid, boundary_conditions=bcs.v)
    w = ZFaceField(grid, boundary_conditions=bcs.w)
    return (u=u, v=v, w=w)
end

function hydrostatic_tendency_fields(velocities, free_surface, grid, tracer_names, bcs)
    u = XFaceField(grid, boundary_conditions=bcs.u)
    v = YFaceField(grid, boundary_conditions=bcs.v)
    tracers = TracerFields(tracer_names, grid, boundary_conditions)
    return merge((u=u, v=v), tracers)
end

function hydrostatic_tendency_fields(velocities, free_surface::ExplicitFreeSurface, grid, tracer_names, bcs)
    u = XFaceField(grid, boundary_conditions=bcs.u)
    v = YFaceField(grid, boundary_conditions=bcs.v)
    η = free_surface_displacement_field(velocities, free_surface, grid)
    tracers = TracerFields(tracer_names, grid, boundary_conditions)
    return merge((u=u, v=v, η=η), tracers)
end

function hydrostatic_tendency_fields(velocities, free_surface::SplitExplicitFreeSurface, grid, tracer_names, bcs)
    u = XFaceField(grid, boundary_conditions=bcs.u)
    v = YFaceField(grid, boundary_conditions=bcs.v)

    U_bcs = barotropic_velocity_boundary_conditions(velocities.u)
    V_bcs = barotropic_velocity_boundary_conditions(velocities.v)
    U = Field{Face, Center, Nothing}(grid, boundary_conditions=U_bcs)
    V = Field{Center, Face, Nothing}(grid, boundary_conditions=V_bcs)

    tracers = TracerFields(tracer_names, grid, boundary_conditions)

    return merge((u=u, v=v, U=U, V=V), tracers)
end

previous_hydrostatic_tendency_fields(::Val{:QuasiAdamsBashforth2}, args...) = hydrostatic_tendency_fields(args...)
previous_hydrostatic_tendency_fields(::Val{:SplitRungeKutta3}, args...) = nothing

function previous_hydrostatic_tendency_fields(::Val{:SplitRungeKutta3}, velocities, free_surface::SplitExplicitFreeSurface, tracername, bcs)
    U_bcs = barotropic_velocity_boundary_conditions(velocities.u)
    V_bcs = barotropic_velocity_boundary_conditions(velocities.v)

    U = Field{Face, Center, Nothing}(grid, boundary_conditions=U_bcs)
    V = Field{Center, Face, Nothing}(grid, boundary_conditions=V_bcs)
    η = free_surface_displacement_field(velocities, free_surface, grid)

    return (; U=U, V=V, η=η)
end
